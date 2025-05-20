from data_preprocess import data_preprocess_ucihar
from data_preprocess import data_preprocess_shar
from data_preprocess import data_preprocess_wisdm
from data_preprocess import data_preprocess_ms
from data_preprocess import data_preprocess_uea
from data_preprocess import data_preprocess_ucr

uea_list = ['ArticularyWordRecognition','AtrialFibrillation','BasicMotions','CharacterTrajectories','Cricket','DuckDuckGeese','EigenWorms','Epilepsy','ERing','EthanolConcentration','FaceDetection','FingerMovements','HandMovementDirection','Handwriting','Heartbeat','JapaneseVowels','Libras','LSST','MotorImagery','NATOPS','PEMS-SF','PenDigits','PhonemeSpectra','RacketSports','SelfRegulationSCP1','SelfRegulationSCP2','SpokenArabicDigits','StandWalkJump','UWaveGestureLibrary','InsectWingbeat']
ucr_list = ['MoteStrain', 'ScreenType', 'MelbournePedestrian', 'RefrigerationDevices', 'PigArtPressure', 'SemgHandSubjectCh2', 'Car', 'HandOutlines', 'NonInvasiveFetalECGThorax2', 'FreezerRegularTrain', 'ArrowHead', 'FreezerSmallTrain', 'ECG200', 'ChlorineConcentration', 'CricketZ', 'CricketX', 'EOGHorizontalSignal', 'DiatomSizeReduction', 'Herring', 'Missing_value_and_variable_length_datasets_adjusted', 'SonyAIBORobotSurface2', 'PickupGestureWiimoteZ', 'ACSF1', 'EOGVerticalSignal', 'Rock', 'FiftyWords', 'ShakeGestureWiimoteZ', 'Symbols', 'ECGFiveDays', 'ProximalPhalanxTW', 'ProximalPhalanxOutlineAgeGroup', 'SyntheticControl', 'Wafer', 'Worms', 'BME', 'MiddlePhalanxTW', 'InsectWingbeatSound', 'UWaveGestureLibraryX', 'Coffee', 'TwoPatterns', 'ShapeletSim', 'Crop', 'AllGestureWiimoteY', 'PigAirwayPressure', 'Meat', 'StarLightCurves', 'UWaveGestureLibraryY', 'PhalangesOutlinesCorrect', 'DistalPhalanxOutlineCorrect', 'Earthquakes', 'CBF', 'Chinatown', 'AllGestureWiimoteZ', 'LargeKitchenAppliances', 'SmoothSubspace', 'GestureMidAirD2', 'MiddlePhalanxOutlineAgeGroup', 'ShapesAll', 'Computers', 'TwoLeadECG', 'DistalPhalanxTW', 'GestureMidAirD3', 'Lightning2', 'ProximalPhalanxOutlineCorrect', 'Plane', 'FacesUCR', 'DodgerLoopGame', 'ItalyPowerDemand', 'CinCECGTorso', 'GunPoint', 'MixedShapesSmallTrain', 'Fungi', 'MiddlePhalanxOutlineCorrect', 'Adiac', 'Phoneme', 'ElectricDevices', 'CricketY', 'NonInvasiveFetalECGThorax1', 'UWaveGestureLibraryZ', 'Yoga', 'BeetleFly', 'Fish', 'ToeSegmentation2', 'MedicalImages', 'Trace', 'GunPointAgeSpan', 'Beef', 'MixedShapesRegularTrain', 'SonyAIBORobotSurface1', 'FaceFour', 'PLAID', 'GesturePebbleZ2', 'OliveOil', 'ToeSegmentation1', 'SemgHandGenderCh2', 'FordB', 'Strawberry', 'Lightning7', 'UWaveGestureLibraryAll', 'InsectEPGSmallTrain', 'SwedishLeaf', 'BirdChicken', 'HouseTwenty', 'FordA', 'DistalPhalanxOutlineAgeGroup', 'InlineSkate', 'SmallKitchenAppliances', 'PigCVP', 'Mallat', 'GestureMidAirD1', 'WormsTwoClass', 'ECG5000', 'GunPointOldVersusYoung', 'Haptics', 'DodgerLoopDay', 'PowerCons', 'EthanolLevel', 'GunPointMaleVersusFemale', 'UMD', 'DodgerLoopWeekend', 'Ham', 'Wine', 'SemgHandMovementCh2', 'FaceAll', 'GesturePebbleZ1', 'AllGestureWiimoteX', 'OSULeaf', 'InsectEPGRegularTrain', 'WordSynonyms', 'MelbournePedestrian', 'PickupGestureWiimoteZ', 'ShakeGestureWiimoteZ', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'GestureMidAirD2', 'GestureMidAirD3', 'DodgerLoopGame', 'PLAID', 'GesturePebbleZ2', 'GestureMidAirD1', 'DodgerLoopDay', 'DodgerLoopWeekend', 'GesturePebbleZ1', 'AllGestureWiimoteX']

def setup_dataloaders(args):
    if args.dataset == 'ucihar':
        args.n_feature = 9
        args.len_sw = 128
        args.n_class = 6
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain == '0'
        train_loaders, val_loader, test_loader = data_preprocess_ucihar.prep_ucihar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int( args.len_sw * 0.5))

    elif args.dataset == 'shar':
        args.n_feature = 3
        args.len_sw = 151
        args.n_class = 17
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain == '1'
        train_loaders, val_loader, test_loader = data_preprocess_shar.prep_shar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5))

    elif args.dataset == 'ms':
        # args.dataset = 'MotionSenseHAR'
        args.n_feature = 12
        args.len_sw = 200
        args.n_class = 6
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain == '1'
        train_loaders, val_loader, test_loader = data_preprocess_ms.prep_ms(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5))

    elif args.dataset == 'wisdm':
        args.n_feature = 3
        args.len_sw = 200
        args.n_class = 6
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain == '1'
        train_loaders, val_loader, test_loader = data_preprocess_wisdm.prep_wisdm(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5))

    if args.dataset in uea_list:
        train_loaders, val_loader, test_loader = data_preprocess_uea.prep_uea(args)

    if args.dataset in ucr_list:
        train_loaders, val_loader, test_loader = data_preprocess_ucr.prep_ucr(args)


    return train_loaders[0], val_loader, test_loader