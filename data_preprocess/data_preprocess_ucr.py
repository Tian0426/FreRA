import os
import numpy as np
import torch
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from data_preprocess.base_loader import base_loader

ucr_list = ['MoteStrain', 'ScreenType', 'MelbournePedestrian', 'RefrigerationDevices', 'PigArtPressure', 'SemgHandSubjectCh2', 'Car', 'HandOutlines', 'NonInvasiveFetalECGThorax2', 'FreezerRegularTrain', 'ArrowHead', 'FreezerSmallTrain', 'ECG200', 'ChlorineConcentration', 'CricketZ', 'CricketX', 'EOGHorizontalSignal', 'DiatomSizeReduction', 'Herring', 'Missing_value_and_variable_length_datasets_adjusted', 'SonyAIBORobotSurface2', 'PickupGestureWiimoteZ', 'ACSF1', 'EOGVerticalSignal', 'Rock', 'FiftyWords', 'ShakeGestureWiimoteZ', 'Symbols', 'ECGFiveDays', 'ProximalPhalanxTW', 'ProximalPhalanxOutlineAgeGroup', 'SyntheticControl', 'Wafer', 'Worms', 'BME', 'MiddlePhalanxTW', 'InsectWingbeatSound', 'UWaveGestureLibraryX', 'Coffee', 'TwoPatterns', 'ShapeletSim', 'Crop', 'AllGestureWiimoteY', 'PigAirwayPressure', 'Meat', 'StarLightCurves', 'UWaveGestureLibraryY', 'PhalangesOutlinesCorrect', 'DistalPhalanxOutlineCorrect', 'Earthquakes', 'CBF', 'Chinatown', 'AllGestureWiimoteZ', 'LargeKitchenAppliances', 'SmoothSubspace', 'GestureMidAirD2', 'MiddlePhalanxOutlineAgeGroup', 'ShapesAll', 'Computers', 'TwoLeadECG', 'DistalPhalanxTW', 'GestureMidAirD3', 'Lightning2', 'ProximalPhalanxOutlineCorrect', 'Plane', 'FacesUCR', 'DodgerLoopGame', 'ItalyPowerDemand', 'CinCECGTorso', 'GunPoint', 'MixedShapesSmallTrain', 'Fungi', 'MiddlePhalanxOutlineCorrect', 'Adiac', 'Phoneme', 'ElectricDevices', 'CricketY', 'NonInvasiveFetalECGThorax1', 'UWaveGestureLibraryZ', 'Yoga', 'BeetleFly', 'Fish', 'ToeSegmentation2', 'MedicalImages', 'Trace', 'GunPointAgeSpan', 'Beef', 'MixedShapesRegularTrain', 'SonyAIBORobotSurface1', 'FaceFour', 'PLAID', 'GesturePebbleZ2', 'OliveOil', 'ToeSegmentation1', 'SemgHandGenderCh2', 'FordB', 'Strawberry', 'Lightning7', 'UWaveGestureLibraryAll', 'InsectEPGSmallTrain', 'SwedishLeaf', 'BirdChicken', 'HouseTwenty', 'FordA', 'DistalPhalanxOutlineAgeGroup', 'InlineSkate', 'SmallKitchenAppliances', 'PigCVP', 'Mallat', 'GestureMidAirD1', 'WormsTwoClass', 'ECG5000', 'GunPointOldVersusYoung', 'Haptics', 'DodgerLoopDay', 'PowerCons', 'EthanolLevel', 'GunPointMaleVersusFemale', 'UMD', 'DodgerLoopWeekend', 'Ham', 'Wine', 'SemgHandMovementCh2', 'FaceAll', 'GesturePebbleZ1', 'AllGestureWiimoteX', 'OSULeaf', 'InsectEPGRegularTrain', 'WordSynonyms', 'MelbournePedestrian', 'PickupGestureWiimoteZ', 'ShakeGestureWiimoteZ', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'GestureMidAirD2', 'GestureMidAirD3', 'DodgerLoopGame', 'PLAID', 'GesturePebbleZ2', 'GestureMidAirD1', 'DodgerLoopDay', 'DodgerLoopWeekend', 'GesturePebbleZ1', 'AllGestureWiimoteX']

class data_loader_ucr(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_ucr, self).__init__(samples, labels, domains)

def load_UCR(dataset,use_fft=False):
    train_file = os.path.join('./data/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('./data/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized data
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        mean = np.nanmean(train)
        std = np.nanstd(train)
        train = (train - mean) / std
        test = (test - mean) / std

    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels

def data_generator(args):
    x_train, y_train, x_test, y_test = load_UCR(args.dataset)

    x_train, x_test = torch.Tensor(x_train), torch.Tensor(x_test)
    if torch.isnan(x_train).sum() > 0:
        x_train = torch.nan_to_num(x_train, nan=0.0)
    if torch.isnan(x_test).sum() > 0:
        x_test = torch.nan_to_num(x_test, nan=0.0)

    unique_y, counts_y = np.unique(y_train, return_counts=True)

    args.n_feature = x_train.shape[-1]
    args.len_sw = x_train.shape[-2]
    args.n_class = len(unique_y)

    train_set_r = data_loader_ucr(x_train, y_train, y_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=True, drop_last=True) # , sampler=sampler)
    val_set_r = data_loader_ucr(x_test, y_test, y_test)
    val_loader_r = DataLoader(val_set_r, batch_size=args.batch_size, shuffle=False)
    test_set_r = data_loader_ucr(x_test, y_test, y_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False)

    return [train_loader_r], None, test_loader_r


def prep_ucr(args):
    if args.cases == 'random':
        return data_generator(args)


