import torch
from torch import nn

class FCN(nn.Module):
    def __init__(self, dataset, n_channels, n_classes, out_channels=128, backbone=True):
        super(FCN, self).__init__()

        self.backbone = backbone

        kernel_size, stride = 8, 1

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=kernel_size, stride=stride, bias=False, padding=int(kernel_size / 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35))
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, stride=stride, bias=False, padding=int(kernel_size / 2)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=kernel_size, stride=stride, bias=False,
                                                   padding=int(kernel_size / 2)),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        if dataset == 'ucihar': # ucihar
            self.out_len = 18
        elif dataset == 'wisdm': # wisdm
            self.out_len = 27
        elif dataset == 'ms': # and n_classes == 6: # ms
            self.out_len = 27
        elif dataset == 'fm':  # fm
            self.out_len = 8
        elif dataset == 'FaceDetection':
            self.out_len = 10
        elif dataset == 'HandMovementDirection':
            self.out_len = 52
        elif dataset == 'Heartbeat':
            self.out_len = 53
        elif dataset == 'Libras':
            self.out_len = 8


        self.out_channels = out_channels
        self.out_dim = self.out_len * self.out_channels

        if backbone == False:
            self.logits = nn.Linear(self.out_dim, n_classes)

    def forward(self, x_in, return_feature=False):
        if len(x_in.shape) == 2:
            x_in = x_in.unsqueeze(-1)
        x_in = x_in.permute(0, 2, 1)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        if self.backbone:
            return x
        else:
            x_flat = x.reshape(x.shape[0], -1)
            logits = self.logits(x_flat)
            if return_feature:
                return logits, x_flat
            else:
                return logits
