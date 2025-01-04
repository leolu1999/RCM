# Raising the Ceiling: Conflict-Free Local Feature Matching with Dynamic View Switching
# RCM Implementation

![Poster](assets/poster.png)
This is a PyTorch implementation of RCM for ECCV'24 [paper](https://arxiv.org/abs/2407.07789), “Raising the Ceiling: Conflict-Free Local Feature Matching with Dynamic View Switching”.

This repo contains training and evaluation scripts used in our paper.

A large part of the code base is borrowed from the [LoFTR](https://github.com/zju3dv/LoFTR) and [ASPanFormer](https://github.com/apple/ml-aspanformer) Repository under their own separate license, terms and conditions. 

## Dependencies
* python 3 >= 3.5
* torch >= 1.1
* opencv >= 3.4
* matplotlib >= 3.1
* numpy >= 1.18
* pytorch-lightning==1.3.5
* torchmetrics==0.6.0 
* kornia
* einops
* loguru
* timm
* PIL

## Data Preparation
Please follow the [training doc](docs/TRAINING.md) for data organization

## MegaDepth Training
To train RCM, run:
```bash
bash scripts/reproduce_train/outdoor.sh
```
To train RCM lite, run:
```bash
bash scripts/reproduce_train/outdoor_lite.sh
```

## Evaluation


### 1. ScanNet Evaluation 
```bash
bash scripts/reproduce_test/indoor.sh
```
```bash
bash scripts/reproduce_test/indoor_lite.sh
```
### 2. MegaDepth Evaluation
 ```bash
bash scripts/reproduce_test/outdoor.sh
```
 ```bash
bash scripts/reproduce_test/outdoor_lite.sh
```

If you find this project useful, please cite:
```
@inproceedings{lu2025raising,
  title={Raising the ceiling: Conflict-free local feature matching with dynamic view switching},
  author={Lu, Xiaoyong and Du, Songlin},
  booktitle={European Conference on Computer Vision},
  pages={256--273},
  year={2025},
}
```
