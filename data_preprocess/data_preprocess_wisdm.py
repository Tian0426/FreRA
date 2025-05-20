# encoding=utf-8
"""
    Created on 10:38 2018/12/17
    @author: Hangwei Qian
    Adapted from: https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs
"""

import os
import numpy as np
import torch
import pickle as cp
from pandas import Series
import zipfile
import argparse
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split, opp_sliding_window_w_d
from sklearn.model_selection import StratifiedShuffleSplit

torch.manual_seed(10)

NUM_FEATURES = 3

class data_loader_wisdm(Dataset):
    def __init__(self, samples, labels, domains):
        self.samples = samples
        self.labels = labels
        self.domains = domains

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain

    def __len__(self):
        return len(self.samples)

def load_domain_data(domain_idx):
    """ to load all the data from the specific domain
    :param domain_idx:
    :return: X and y data of the entire domain
    """
    data_dir = './data/WISDM_ar_v1.1/'
    saved_filename = 'wisdm_domain_' + domain_idx + '_wd.data'
    if os.path.isfile(data_dir + saved_filename) == True:
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X = data[0][0]
        y = data[0][1]
        d = data[0][2]
    else:
        str_folder = './data/WISDM_ar_v1.1/'
        data_all = np.genfromtxt(str_folder + 'WISDM_ar_v1.1_raw_hangwei_v2.txt', delimiter=',', usecols=[0,1,3,4,5])

        X_all = data_all[:, 2:] # data: (1098209, 3)
        y_all = data_all[:, 1] - 1 # to map the labels from [1,...,6] to [0, 5]
        id_all = data_all[:, 0]


        print('\nProcessing domain {0} files...\n'.format(domain_idx))
        target_idx = np.where(id_all == int(domain_idx))
        X = X_all[target_idx]
        y = y_all[target_idx]

        # change the domain index from string ['1','2','3','4',...]to [0,1,2,3,4]
        # todo if further change
        domain_idx_map = {'1':0, '3':1, '5':2, '6':3, '7':4, '8':5,
                          '12':6, '13':7, '18':8, '19':9, '20':10,
                          '21':11, '24':12, '27':13, '29':14,
                          '31':15, '32':16, '33':17, '34':18, '36':19}
        # domain_idx_now = int(domain_idx[-1])
        # if domain_idx_now < 10:
        #     domain_idx_int = domain_idx_now - 1
        # else:
        #     domain_idx_int = domain_idx_now - 1
        domain_idx_int = domain_idx_map[domain_idx]

        d = np.full(y.shape, domain_idx_int, dtype=int)
        print('\nProcessing domain {0} files | X: {1} y: {2} d:{3} \n'.format(domain_idx, X.shape, y.shape, d.shape))

        obj = [(X, y, d)]
        # file is not supported in python3, use open instead, by hangwei
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
    return X, y, d



def prep_domains_wisdm_subject(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):

    # hangwei: for wisdm data, total domains is [1,..., 36]
    #   complete data domains     source_domain_list = ['1', '3', '5','6','7','8',
    #                            '12', '13','18','19','20',
    #                           '21', '24','27', '29',
    #                           '31', '32', '33', '34','36']

    # source_domain_list = ['1', '2', '3', '4', '5','6','7','8','9','10',
    #                       '11', '12', '13', '14', '15','16','17','18','19','20',
    #                       '21', '22', '23', '24', '25','26','27','28','29','30',
    #                       '31', '32', '33', '34', '35','36']
    source_domain_list = ['1', '3', '5', '6', '7', '8',
                          '12', '13', '18', '19', '20',
                          '21', '24', '27', '29',
                          '31', '32', '33', '34', '36']
    source_domain_list.remove(args.target_domain)

    # source domain data prep
    source_loaders = []
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)
        y = y.astype(int)
        x_win, y_win, d_win = opp_sliding_window_w_d(x, y, d, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        print(" ..after sliding window: inputs {0}, targets {1}".format(x_win.shape, y_win.shape))
        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win

    # get the info of the dataset, by hangwei. 1.15.2019
    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()

    # updated by hangwei: sample_weights = weights[y_win]
    sample_weights = get_sample_weights(y_win, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)
    data_set = data_loader_wisdm(x_win, y_win, d_win)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    print('source_loader batch: ', len(source_loader))
    source_loaders.append(source_loader)

    # target domain data prep
    print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)
    y = y.astype(int)
    x_win, y_win, d_win = opp_sliding_window_w_d(x, y, d, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    print(" ..after sliding window: inputs {0}, targets {1}".format(x_win.shape, y_win.shape))

    data_set = data_loader_wisdm(x_win, y_win, d_win)
    # padsequence() to deal with varying length input of each data example
    # shuffle is forced to be False when sampler is available
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    print('target_loader batch: ', len(target_loader))

    return source_loaders, None, target_loader

def prep_domains_wisdm_subject_small(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):

    # hangwei: for wisdm data, total domains is [1,..., 36]
    #   complete data domains     source_domain_list = ['1', '3', '5','6','7','8',
    #                            '12', '13','18','19','20',
    #                           '21', '24','27', '29',
    #                           '31', '32', '33', '34','36']

    # source_domain_list = ['1', '2', '3', '4', '5','6','7','8','9','10',
    #                       '11', '12', '13', '14', '15','16','17','18','19','20',
    #                       '21', '22', '23', '24', '25','26','27','28','29','30',
    #                       '31', '32', '33', '34', '35','36']
    source_domain_list = ['20', '31', '8', '12', '13']
    source_domain_list.remove(args.target_domain)

    # source domain data prep
    source_loaders = []
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)
        y = y.astype(int)
        x_win, y_win, d_win = opp_sliding_window_w_d(x, y, d, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        print(" ..after sliding window: inputs {0}, targets {1}".format(x_win.shape, y_win.shape))
        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win

    # get the info of the dataset, by hangwei. 1.15.2019
    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()

    # updated by hangwei: sample_weights = weights[y_win]
    sample_weights = get_sample_weights(y_win, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)
    data_set = data_loader_wisdm(x_win, y_win, d_win)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    print('source_loader batch: ', len(source_loader))
    source_loaders.append(source_loader)

    # target domain data prep
    print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)
    y = y.astype(int)
    x_win, y_win, d_win = opp_sliding_window_w_d(x, y, d, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    print(" ..after sliding window: inputs {0}, targets {1}".format(x_win.shape, y_win.shape))

    data_set = data_loader_wisdm(x_win, y_win, d_win)
    # padsequence() to deal with varying length input of each data example
    # shuffle is forced to be False when sampler is available
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    print('target_loader batch: ', len(target_loader))

    return source_loaders, None, target_loader



def prep_domains_wisdm_random(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):

    # hangwei: for wisdm data, total domains is [1,..., 36]
    #   complete data domains     source_domain_list = ['1', '3', '5','6','7','8',
    #                            '12', '13','18','19','20',
    #                           '21', '24','27', '29',
    #                           '31', '32', '33', '34','36']
    # source_domain_list = ['1', '2', '3', '4', '5','6','7','8','9','10',
    #                       '11', '12', '13', '14', '15','16','17','18','19','20',
    #                       '21', '22', '23', '24', '25','26','27','28','29','30',
    #                       '31', '32', '33', '34', '35','36']
    source_domain_list = ['1', '3', '5', '6', '7', '8',
                          '12', '13', '18', '19', '20',
                          '21', '24', '27', '29',
                          '31', '32', '33', '34', '36']
    # source_domain_list.remove(args.target_domain)

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, split_ratio = [], 0, 0.0

    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)
        y = y.astype(int)
        x_win, y_win, d_win = opp_sliding_window_w_d(x, y, d, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        print(" ..after sliding window: inputs {0}, targets {1}".format(x_win.shape, y_win.shape))

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win

        n_train.append(x_win.shape[0])

        unique_y, counts_y = np.unique(y_win_all, return_counts=True)
        print('y_train label distribution: ', dict(zip(unique_y, counts_y)))

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_split(x_win_all, y_win_all, d_win_all,
                                                              split_ratio=args.split_ratio)

    print('x_win_train', x_win_train.shape)
    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)
    train_set_r = data_loader_wisdm(x_win_train, y_win_train, d_win_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size,shuffle=False, drop_last=True, sampler=sampler)
    val_set_r = data_loader_wisdm(x_win_val, y_win_val, d_win_val)
    val_loader_r = DataLoader(val_set_r, batch_size=args.batch_size, shuffle=False)
    test_set_r = data_loader_wisdm(x_win_test, y_win_test, d_win_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False)

    return [train_loader_r], val_loader_r, test_loader_r


def prep_wisdm(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    if args.cases == 'subject':
        return prep_domains_wisdm_subject(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    if args.cases == 'subject_small':
        return prep_domains_wisdm_subject_small(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'random':
        return prep_domains_wisdm_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    else:
        return 'Error!\n'
