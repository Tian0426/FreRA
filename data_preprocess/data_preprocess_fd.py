import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from data_preprocess.base_loader import base_loader
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from sklearn.model_selection import StratifiedShuffleSplit

def load_domain_data(domain_idx):
    data_dir = './data/FD/'
    filename = domain_idx +'.pt'
    print(filename)
    if os.path.isfile(data_dir + filename) == True:
        data = torch.load(data_dir + filename)
        x = data['x']
        y = data['y']
    else:
        for domain in ['a', 'b', 'c', 'd']:
            all_x, all_y = None, None
            for pre in ['train', 'val', 'test']:
                filename = pre + '_' + domain + '.pt'
                data = torch.load('./data/FD/' + filename)
                x = data['samples']
                y = data['labels']
                print(filename, x.shape, y.shape)
                all_x = torch.cat([all_x, x], axis=0) if all_x is not None else x
                all_y = torch.cat([all_y, y], axis=0) if all_y is not None else y
            unique_y, counts_y = np.unique(all_y, return_counts=True)
            # print(x[0, :10])
            print(all_x.shape, all_y.shape)
            print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
            torch.save({'x': all_x, 'y': all_y}, './data/FD/' + domain + '.pt')
            data = torch.load(data_dir + domain + '.pt')
            x = data['x']
            y = data['y']
    # print({'a': 0, 'b': 1, 'c': 2, 'd': 3}[domain_idx])
    d = torch.Tensor(np.full(y.shape, {'a': 0, 'b': 1, 'c': 2, 'd': 3}[domain_idx], dtype=int))
    print(x.shape, y.shape, d.shape)
    unique_y, counts_y = np.unique(y, return_counts=True)
    print('y label distribution: ', dict(zip(unique_y, counts_y)))
    return x, y, d

def load_domain_data_bd(domain_idx='bd'):
    if domain_idx != 'bd':
        return 'Error! Domain idx should be bd\n'
    data_dir = './data/FD/'
    filename = domain_idx + '.pt'
    if os.path.isfile(data_dir + filename) == True:
        data = torch.load(data_dir + filename)
        x = data['x']
        y = data['y']
    else:
        all_x, all_y = None, None
        for domain in ['b', 'd']:
            filename = domain +'.pt'
            print(filename)
            if os.path.isfile(data_dir + filename) == True:
                data = torch.load(data_dir + filename)
                x = data['x']
                y = data['y']
                sp = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
                for selected_index, _ in sp.split(x, y):
                    x_selected, y_selected = x[selected_index], y[selected_index]
                all_x = torch.cat([all_x, x_selected], axis=0) if all_x is not None else x_selected
                all_y = torch.cat([all_y, y_selected], axis=0) if all_y is not None else y_selected
        unique_y, counts_y = np.unique(all_y, return_counts=True)
        print(all_x.shape, all_y.shape)
        print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
        torch.save({'x': all_x, 'y': all_y}, './data/FD/' + domain_idx + '.pt')
        data = torch.load(data_dir + domain_idx + '.pt')
        x = data['x']
        y = data['y']
    d = torch.Tensor(np.full(y.shape, {'a': 0, 'bd': 1, 'c': 2}[domain_idx], dtype=int))
    print(x.shape, y.shape, d.shape)
    unique_y, counts_y = np.unique(y, return_counts=True)
    print('y label distribution: ', dict(zip(unique_y, counts_y)))
    return x, y, d

class data_loader_fd(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_fd, self).__init__(samples, labels, domains)

def prep_domains_fd_comb(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    # note: for fd dataset with total 4 domains,
    source_domain_list = ['a', 'bd', 'c']

    source_domain_list.remove(args.target_domain)

    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)

        if source_domain == 'bd':
            x, y, d = load_domain_data_bd(source_domain)
        else:
            x, y, d = load_domain_data(source_domain)

        x = x.reshape(-1, 5120, 1)
        print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()

    sample_weights = get_sample_weights(y_win_all, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    data_set = data_loader_fd(x_win_all, y_win_all, d_win_all)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    print('source_loader batch: ', len(source_loader))
    source_loaders = [source_loader]

    # target domain data prep
    print('target_domain:', args.target_domain)
    if args.target_domain == 'bd':
        x, y, d = load_domain_data_bd(args.target_domain)
    else:
        x, y, d = load_domain_data(args.target_domain)

    x = x.reshape(-1, 5120, 1)

    print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

    data_set = data_loader_fd(x, y, d)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    print('target_loader batch: ', len(target_loader))
    return source_loaders, None, target_loader

def prep_domains_fd_random(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    # note: for fd dataset with total 4 domains,
    source_domain_list = ['a', 'b', 'd', 'c']

    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)

        if source_domain == 'bd':
            x, y, d = load_domain_data_bd(source_domain)
        else:
            x, y, d = load_domain_data(source_domain)

        x = x.reshape(-1, 5120, 1)
        print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_split(x_win_all, y_win_all, d_win_all,
                                                              split_ratio=args.split_ratio)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_set_r = data_loader_fd(x_win_train, y_win_train, d_win_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    val_set_r = data_loader_fd(x_win_val, y_win_val, d_win_val)
    val_loader_r = DataLoader(val_set_r, batch_size=args.batch_size, shuffle=False)
    test_set_r = data_loader_fd(x_win_test, y_win_test, d_win_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False)

    return [train_loader_r], val_loader_r, test_loader_r

def prep_eeg(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    if args.cases == 'rich_comb':
        return prep_domains_fd_comb(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    if args.cases == 'random':
        return prep_domains_fd_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == '':
        pass
    else:
        return 'Error!\n'
