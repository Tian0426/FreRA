import argparse
import math
import os
import shutil
import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import models.builder
from data_loaders import *
from augmentations import *
from models.backbones import FCN

import fitlog
from sklearn.metrics import f1_score
from copy import deepcopy
from autoaug.fourier import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# dataset
parser.add_argument('--dataset', type=str, default='ucihar', help='name of dataset')
parser.add_argument('--n_feature', type=int, default=77, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=30, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=18, help='number of class')
parser.add_argument('--cases', type=str, default='random',
                    choices=['random', 'subject', 'large_subject'], help='name of scenarios')
parser.add_argument('--split_ratio', type=float, default=0.2,
                    help='split ratio of test/val: train(0.64), val(0.16), test(0.2)')
parser.add_argument('--target_domain', type=str, default='0')

parser.add_argument('--framework', type=str, default='simclr',
                    choices=['simclr', 'byol', 'simsiam'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='FCN',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: FCN)')
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--f_lr', default=0.01, type=float, help='initial learning rate for fourier weight')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--low_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')

parser.add_argument('--cos', action='store_true', default=True,
                    help='use cosine lr schedule')

parser.add_argument('--logdir', default='log', type=str,
                    help='fitlog directory')

parser.add_argument('--f_temperature', default=0.1, type=float,
                    help='temperature for Fourier AutoAug')
parser.add_argument('--l1_weight', default=0.1, type=float,
                    help='weight of l1-norm of f_aug weight para')
parser.add_argument('--f_aug_mode', default='FreRA', type=str,)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    best_model = main_worker(args.gpu, args)

    main_worker_cls(args.gpu, best_model, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, eval_loader = setup_dataloaders(args)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.framework == 'simclr':
        model = models.builder.SimCLR(
            DEVICE,
            args.dataset,
            args.n_feature,
            args.batch_size,
            args.arch,
            args.low_dim, args.temperature)
    elif args.framework == 'byol':
        model = models.builder.BYOL(
            DEVICE,
            args.arch,
            args.dataset,
            args.n_feature,
            args.len_sw,
            moving_average=0.996)
    elif args.framework == 'simsiam':
        model = models.builder.BYOL(
            DEVICE,
            args.arch,
            args.dataset,
            args.n_feature,
            args.len_sw,
            moving_average=0.0)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.to(DEVICE)

    # define loss function (criterion) and optimizer
    if args.framework in ['simclr']:
        criterion = nn.CrossEntropyLoss().to(DEVICE)
    elif args.framework in ['byol', 'simsiam']:
        criterion = nn.CosineSimilarity(dim=1)

    if args.framework in ['simclr']:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.framework in ['byol', 'simsiam']:
        if args.framework == 'byol':
            args.weight_decay = 1.5e-6
            lr_mul = 10.0
        elif args.framework == 'simsiam':
            args.weight_decay = 1e-4
            lr_mul = 1.0
        optimizer1 = torch.optim.Adam(model.encoder_q.parameters(),
                                      args.lr,
                                      weight_decay=args.weight_decay)
        optimizer2 = torch.optim.Adam(model.online_predictor.parameters(),
                                      args.lr * lr_mul,
                                      weight_decay=args.weight_decay)
        optimizer = [optimizer1, optimizer2]

    cudnn.benchmark = True

    # fitlog
    if os.path.isdir(args.logdir) == False:
        os.makedirs(args.logdir)
    fitlog.set_log_dir(args.logdir)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)

    # autoaug
    if args.f_aug_mode == 'FreRA':
        aug_f = FreRA(len_sw=args.len_sw, device=DEVICE).to(DEVICE)

    f_optimizer = torch.optim.AdamW(aug_f.parameters(), lr=args.f_lr)

    f_weight = []

    for epoch in range(args.start_epoch, args.epochs):

        if args.framework not in ['byol', 'simsiam']:
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(epoch, aug_f, f_optimizer, DEVICE, train_loader, model, criterion, optimizer, args, fitlog)

        f_weight.append(aug_f.weight.cpu().detach().numpy())

    # save weights
    fitlog.add_hyper(aug_f.weight, name='fourier weight')

    return deepcopy(model.state_dict())

def train(epoch, aug_f, f_optimizer, DEVICE, train_loader, model, criterion, optimizer, args, fitlog=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    l1_losses = AverageMeter('L1Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (sample, target, domain) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        sample = sample.to(DEVICE)
        aug_sample1, aug_sample2 = aug_f(sample, temperature=args.f_temperature).float(), gen_aug(None, sample, 'na').to(DEVICE).float()

        # compute output
        if args.framework in ['simclr']:
            output, target, z1, z2 = model(im_q=aug_sample1, im_k=aug_sample2)
            loss = criterion(output, target)
        elif args.framework in ['byol', 'simsiam']:
            online_pred_one, online_pred_two, target_proj_one, target_proj_two = model(im_q=aug_sample1, im_k=aug_sample2)
            loss = -(criterion(online_pred_one, target_proj_two).mean() + criterion(online_pred_two, target_proj_one).mean()) * 0.5

        # l1-norm loss of weight para
        l1_weight_loss = torch.norm(aug_f.para[:, 0], p=1)
        loss = loss + l1_weight_loss * args.l1_weight / args.len_sw

        losses.update(loss.item(), aug_sample1.size(0))
        l1_losses.update(l1_weight_loss.item(), aug_sample1.size(0))

        # update cl framework and fourier weight
        if args.framework in ['simclr']:
            optimizer.zero_grad()
            f_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            f_optimizer.step()
        elif args.framework in ['byol', 'simsiam']:
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            f_optimizer.zero_grad()
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()
            f_optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    fitlog.add_loss(losses.avg, name="InfoNCE loss", step=epoch)
    fitlog.add_loss(l1_losses.avg, name="L1 loss", step=epoch)
    fitlog.add_metric({"dev": {"Inst Acc": acc_inst.avg}}, step=epoch)

    print(
        f'epoch {epoch}    InfoNCE loss     : {losses.avg:.4f},   L1 loss     : {l1_losses.avg:.4f}')

    progress.display(i)

def main_worker_cls(gpu, best_model, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = FCN(args.dataset, n_channels=args.n_feature, n_classes=args.n_class, backbone=False)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['logits.weight', 'logits.bias']:
            param.requires_grad = False
    # init the fc layer
    model.logits.weight.data.normal_(mean=0.0, std=0.01)
    model.logits.bias.data.zero_()

    # load best model
    # rename pre-trained keys
    state_dict = deepcopy(best_model)
    for k in list(state_dict.keys()):
        if 'net' in k:
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder_q.net') and not k.startswith('encoder_q.net.logits'):
                # remove prefix
                state_dict[k[len("encoder_q.net."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        else:
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder_q') and not k.startswith('encoder_q.logits'):
                # remove prefix
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    args.start_epoch = 0
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"logits.weight", "logits.bias"}

    print("=> loaded pre-trained model ")

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.to(DEVICE)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    train_loader, val_loader, test_loader = setup_dataloaders(args)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_cls(DEVICE, train_loader, val_loader, model, criterion, optimizer, epoch, args)

        if epoch == args.start_epoch:
            sanity_check(model.state_dict(), best_model)
    acc1 = validate_cls(DEVICE, test_loader, model, criterion, args, epoch, val=False)


def train_cls(DEVICE, train_loader, val_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (sample, target, domain) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        sample = sample.to(DEVICE).float()
        target = target.to(DEVICE).long()

        # compute output
        output = model(sample)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        # print(acc1)
        losses.update(loss.item(), sample.size(0))
        top1.update(acc1[0].item(), sample.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    fitlog.add_loss(losses.avg, name="CLS Train loss", step=epoch)
    fitlog.add_loss(optimizer.param_groups[0]['lr'], name="CLS lr", step=epoch)
    fitlog.add_metric({"dev": {"CLS Train Acc": top1.avg}}, step=epoch)

    progress.display(i)

    if val_loader is not None:
        acc1_val = validate_cls(DEVICE, val_loader, model, criterion, args, epoch)


def validate_cls(DEVICE, val_loader, model, criterion, args, epoch, val=True):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    total = 0
    correct = 0
    trgs = np.array([])
    preds = np.array([])
    feats = None
    confusion_matrix = torch.zeros(args.n_class, args.n_class)

    with torch.no_grad():
        end = time.time()
        for i, (sample, target, domain) in enumerate(val_loader):
            sample = sample.to(DEVICE).float()
            target = target.to(DEVICE).long()

            # compute output
            output, feat = model(sample, return_feature=True)
            loss = criterion(output, target)

            if not val:
                _, predicted = torch.max(output.data, 1)
                trgs = np.append(trgs, target.data.cpu().numpy())
                preds = np.append(preds, predicted.data.cpu().numpy())
                if feats is None:
                    feats = feat
                else:
                    feats = torch.cat((feats, feat), 0)
                for t, p in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                total += target.size(0)
                correct += (predicted == target).sum()

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), sample.size(0))
            top1.update(acc1[0].item(), sample.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if val:
            fitlog.add_loss(losses.avg, name="CLS Val loss", step=epoch)
            fitlog.add_metric({"dev": {"CLS Val Acc": top1.avg}}, step=epoch)

        if not val:
            acc_test = float(correct) * 100.0 / total
            miF = f1_score(trgs, preds, average='micro') * 100
            maF = f1_score(trgs, preds, average='macro') * 100

            fitlog.add_best_metric({"dev": {"Test Acc": acc_test}})
            fitlog.add_best_metric({"dev": {"miF": miF}})
            fitlog.add_best_metric({"dev": {"maF": maF}})
            fitlog.add_hyper(confusion_matrix, name='conf_mat')

        progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} '
              .format(top1=top1))
    return top1.avg


def sanity_check(state_dict, best_model):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading model for sanity check")
    state_dict_pre = best_model

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'logits.weight' in k or 'logits.bias' in k:
            continue
        # name in pretrained model
        k_pre = 'encoder_q.' + k

        if 'net' in list(state_dict_pre.keys())[0]:
            k_pre = 'encoder_q.net.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre].cpu()).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    print(filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        for meter in self.meters:
            print(meter)
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_cls(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()