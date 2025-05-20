import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from data_preprocess.base_loader import base_loader

class data_loader_uea(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_uea, self).__init__(samples, labels, domains)

def data_generator(args):
    data_path = './data/'+args.dataset+'/'

    train_dataset = torch.load(os.path.join(data_path, args.dataset+"_train.pt"))
    x_train, y_train = train_dataset["samples"], train_dataset["labels"]
    test_dataset = torch.load(os.path.join(data_path, args.dataset+"_test.pt"))
    x_test, y_test = test_dataset["samples"], test_dataset["labels"]

    if isinstance(x_train, np.ndarray):
        x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
        y_train, y_test = torch.from_numpy(y_train).long(), torch.from_numpy(y_test).long()

    d_train = np.full(x_train.shape[0], 0)
    d_test = np.full(x_test.shape[0], 0)

    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test))
    d_all = np.concatenate((d_train, d_test))

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_split(x_all, y_all, d_all, split_ratio=args.split_ratio)

    print(x_win_train.shape, x_win_val.shape, x_win_test.shape)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                                             replacement=True)

    train_set_r = data_loader_uea(x_win_train, y_win_train, d_win_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    val_set_r = data_loader_uea(x_win_val, y_win_val, d_win_val)
    val_loader_r = DataLoader(val_set_r, batch_size=args.batch_size, shuffle=False)
    test_set_r = data_loader_uea(x_test, y_test, d_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False)

    return [train_loader_r], val_loader_r, test_loader_r

def prep_uea(args):
    if args.cases == 'random':
        return data_generator(args)