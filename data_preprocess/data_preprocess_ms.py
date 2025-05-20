'''
# Data Pre-processing on UEA datasets.
#
# '''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pickle as cp
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split, opp_sliding_window, normalize
from data_preprocess.base_loader import base_loader


class data_loader_uea(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_uea, self).__init__(samples, labels, domains)

def apply_label_map(y, label_map):
    y_mapped = []
    for l in y:
        y_mapped.append(label_map.get(l))
    return np.array(y_mapped)

def get_windows_dataset_from_user_list_format(user_datasets, window_size=200, shift=100):
    user_dataset_windowed = {}
    label_list = ['sit', 'std', 'wlk', 'ups', 'dws', 'jog'] # no null class
    label_map = dict([(l, i) for i, l in enumerate(label_list)])

    for user_id in user_datasets:
        x = []
        y = []

        # Loop through each trail of each user
        for v, l in user_datasets[user_id]:
            # print(l)
            l = apply_label_map(l, label_map)
            # print(l)
            v_sw, l_sw = opp_sliding_window(v, l, window_size, shift)

            if len(v_sw) > 0:
                x.append(v_sw)
                y.append(l_sw)
            # print(f"Data: {v_sw.shape}, Labels: {l_sw.shape}")

        # combine all trials
        user_dataset_windowed[user_id] = (np.concatenate(x), np.concatenate(y).squeeze())

    x = []
    y = []
    d = []
    for user_id in user_dataset_windowed:

        v, l = user_dataset_windowed[user_id]
        x.append(v)
        y.append(l)
        d.append(np.full(len(l), user_id))

    x = np.concatenate(x)
    y = np.concatenate(y).squeeze()
    d = np.concatenate(d).squeeze()

    return x, y, d

def prep_ms_random(args, sw, ss):
    # with open('data/MotionSense/motion_sense_user_split.pkl', 'rb') as f:
    with open('data/MotionSense/A_DeviceMotion_data/motion_sense_user_split.pkl', 'rb') as f:
        dataset_dict = cp.load(f)
        user_datasets = dataset_dict['user_split']

    x, y, d = get_windows_dataset_from_user_list_format(user_datasets, window_size=sw, shift=ss)
    x = normalize(x)
    print(x.shape, y.shape, d.shape)

    x_train, x_val, x_test, \
    y_train, y_val, y_test, \
    d_train, d_val, d_test = train_test_val_split(x, y, d, split_ratio=args.split_ratio)

    unique_y, counts_y = np.unique(y_train, return_counts=True)
    _, dataset_len_sw, dataset_n_feature = x_train.shape
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                                             replacement=True)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0, 0, 0, 0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1, 1, 1, 1))
    # ])

    print(y_train.shape, y_val.shape, y_test.shape)

    print(x_train.shape, x_val.shape, x_test.shape)
    train_set_r = data_loader_uea(x_train, y_train, d_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    val_set_r = data_loader_uea(x_val, y_val, d_val)
    val_loader_r = DataLoader(val_set_r, batch_size=args.batch_size, shuffle=False)
    test_set_r = data_loader_uea(x_test, y_test, d_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False)

    return [train_loader_r], val_loader_r, test_loader_r




def prep_ms(args, SLIDING_WINDOW_LEN=200, SLIDING_WINDOW_STEP=100):
    # todo: to check whether uea dataset belongs to subject or random
    if args.cases == 'random':
        return prep_ms_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    # elif args.cases == 'subject':
    #     return prep_domains_ucihar_subject(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    # elif args.cases == 'subject_large':
    #     return prep_domains_ucihar_subject_large(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'