## FreRA: A Frequency-Refined Augmentation for Contrastive Learning on Time Series Classification

Code for KDD 2025 paper "FreRA: A Frequency-Refined Augmentation for Contrastive Learning on Time Series Classification"

## Abstract

>Contrastive learning has emerged as a competent approach for unsupervised representation learning. However, the design of an optimal augmentation strategy, although crucial for contrastive learning, is less explored for time series classification tasks. Existing predefined time-domain augmentation methods are primarily adopted from vision and are not specific to time series data. Consequently, this cross-modality incompatibility may distort the semantically relevant information of time series by introducing mismatched patterns into the data. To address this limitation, we present a novel perspective from the frequency domain and identify three advantages for downstream classification: 1) the frequency component naturally encodes global features, 2) the orthogonal nature of the Fourier basis allows easier isolation and independent modifications of critical and unimportant information, and 3) a compact set of frequency components can preserve semantic integrity. To fully utilize the three properties, we propose the lightweight yet effective Frequency Refined Augmentation (FreRA) tailored for time series contrastive learning on classification tasks, which can be seamlessly integrated with contrastive learning frameworks in a plug-and-play manner. Specifically, FreRA automatically separates critical and unimportant frequency components. Accordingly, we propose semantic-aware Identity Modification and semantic-agnostic Self-adaptive Modification to protect semantically relevant information in the critical frequency components and infuse variance into the unimportant ones respectively. Theoretically, we prove that FreRA generates semantic-preserving views. Empirically, we conduct extensive experiments on two benchmark datasets including UCR and UEA archives, as well as five large-scale datasets on diverse applications. FreRA consistently outperforms ten leading baselines on time series classification, anomaly detection, and transfer learning tasks, demonstrating superior capabilities in contrastive representation learning and generalization in transfer learning scenarios across diverse datasets.

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
