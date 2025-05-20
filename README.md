## FreRA: A Frequency-Refined Augmentation for Contrastive Learning on Time Series Classification (KDD-2025)

## Environment Setup
Build an environment with Anaconda to install required packages 
```
conda create -n FreRA python=3.8.3
conda activate FrerA
pip install -r requirements.txt
```

## Models
The following models are provided under `./models/`. 
- contrastive models: SimCLR, BYOL ```./models/builder.py```
- backbone encoder: FCN ```./models/backbones.py```

## Main Functions
- ```main_FreRA.py``` 
  
## Datasets 
Datasets can be downloaded from the following websites to folder `./data/` and the datasets will be pre-processed automatically by our codes under `./data_preprocess/`. 
- [UCIHAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php)
- [MotionSense](https://github.com/mmalekzadeh/motion-sense)
- [SHAR](http://www.sal.disco.unimib.it/technologies/unimib-shar/)
- [Fault Diagnosis](https://mb.uni-paderborn.de/kat/datacenter)
- [UEA Archive](https://timeseriesclassification.com/dataset.php)
- [UCR Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
  
## Usage
Scripts for using the proposed FreRA model.
```angular2html
python main_FreRA.py --f_aug_mode 'FreRA' --l1_weight 0.003 --framework 'simclr' --dataset 'ucihar' --lr 0.01 --f_lr 0.001  --batch_size 128 --epochs 200 --temperature 0.2 --f_temperature 0.1 --gpu 0
python main_FreRA.py --f_aug_mode 'FreRA' --l1_weight 0.003 --framework 'simclr' --dataset 'wisdm' --lr 0.01 --f_lr 0.001  --batch_size 128 --epochs 200 --temperature 0.2 --f_temperature 0.1 --gpu 0
python main_FreRA.py --f_aug_mode 'FreRA' --l1_weight 0.003 --framework 'simclr' --dataset 'ms' --lr 0.01 --f_lr 0.001  --batch_size 128 --epochs 200 --temperature 0.2 --f_temperature 0.1 --gpu 0
```
