
[![Travis](https://img.shields.io/badge/language-Python-red.svg)]()

[![GitHub stars](https://img.shields.io/github/stars/murufeng/EPSANet.svg?style=social&label=Stars)](https://github.com/murufeng/EPSANet)
[![GitHub forks](https://img.shields.io/github/forks/murufeng/EPSANet.svg?style=social&label=Forks)](https://github.com/murufeng/EPSANet)


This repo contains the official Pytorch implementaion code for MicroDepthNet.


## Installation

### Requirements

- Python 3.9
- PyTorch

### GPUV100 environments

- OS: Ubuntu 18.04
- CUDA: 11.4
- Toolkit: PyTorch
- GPU: two V100

### GPU4090 environments

- OS: Ubuntu 18.04
- CUDA: 11.4
- Toolkit: PyTorch
- GPU: 4090

## Image-net Data preparation

Download and extract ImageNet train and val images from [http://image-net.org/](http://image-net.org/).
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## CIFAR10/100 Data preparation

Using `--dataset + "cifar10/cifar100"` switch training dataset.
```
/path/to/cifar10\100/
  train/
    
  val/
    
```
# Usage
First, clone the repository locally:
```
git clone https://github.com/SimonHanYANG/MicroDepthNet.git
cd MicroDepthNet
```
- Create a conda virtual environment and activate it:

```bash
conda create -n microdepthnet python=3.9
conda activate microdepthnet
```

- Install `pytorch` using `pip` for GPUV100 Environment

```
# CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

- Install `pytorch` using `pip` for GPU4090 Environment

```
# CUDA 12.0
pip install torch torchvision torchaudio
```
## Training
To train models on CIFAR with 8 gpus run:

```
-a Arch/Model Name
-b Batch Size
--dataset Category of using dataset
--action Other Action (for log root name suffix)
--data Download Dataset in this FilePath
```
```
CUDA_VISIBLE_DEVICES=0,1 python main.py -a epsanet50 -b 256 --dataset cifar100 --action cifar100 --data ./dataset_cifar100
```

## Traing Visualization
Using tensorboard for visualization:

```
tensorboard --logdir /path_to_log_root/ --bind_all
```
- SSH connect to remote server:
```
ssh -L 16006:127.0.0.1:6006 cuhk
```
- Open visualization URL:
```
http://127.0.0.1:16006/
```

## Backbone EPSANet Pretrained Model Zoo

Models are trained with 8 GPUs on both ImageNet and MS-COCO 2017 dataset. 

### Image Classification on ImageNet

|         Model         | Params(M) | FLOPs(G) | Top-1 (%) | Top-5 (%) | 
|:---------------------:|:---------:|:--------:|:---------:|:---------:|
| EPSANet-50(Small)             |  22.56     | 3.62     | 77.49 | 93.54 |
| EPSANet-50(Large)             | 27.90     | 4.72    | 78.64 | 94.18 | 
| EPSANet-101(Small)             | 38.90   | 6.82     | 78.43 | 94.11 | 
| EPSANet-101(Large)            | 49.59     | 8.97    | 79.38 | 94.58  |


### Object Detection on MS-COCO 2017

#### Faster R-CNN
|    model |  Style  | Lr schd | Params(M) | FLOPs(G) | box AP  | AP_50  |  AP_75| 
| :-------------:| :-----: | :-----: |:---------:|:--------:| :----: | :--------: | :----: | 
|    EPSANet-50(small)  | pytorch |   1x    | 38.56 | 197.07 | 39.2 | 60.3 | 42.3 | 
|    EPSANet-50(large)  | pytorch |   1x    | 43.85 | 219.64 | 40.9 | 62.1 | 44.6 | 


#### Mask R-CNN
|    model |  Style  | Lr schd | Params(M) | FLOPs(G) | box AP | AP_50  |  AP_75  | 
| :-------------:| :-----: | :-----: |:---------:|:--------:| :----: | :------: | :----: | 
|    EPSANet-50(small)  | pytorch |   1x    | 41.20 | 248.53 | 40.0 | 60.9 | 43.3 | 
|    EPSANet-50(large)  | pytorch |   1x    | 46.50 | 271.10 | 41.4 | 62.3 | 45.3 | 

#### RetinaNet
|    model |  Style  | Lr schd | Params(M) | FLOPs(G) | box AP | AP_50  |  AP_75  | 
| :-------------:| :-----: | :-----: |:---------:|:--------:| :----: | :------: | :----: | 
|    EPSANet-50(small)  | pytorch |   1x    | 34.78 | 229.32 | 38.2  | 58.1 | 40.6 | 
|    EPSANet-50(large)  | pytorch |   1x    | 40.07 | 251.89 | 39.6  | 59.4 | 42.3 | 


### Instance segmentation with Mask R-CNN on MS-COCO 2017
|model |Params(M) | FLOPs(G) | AP | AP_50 | AP_75 | 
| :----:| :-----: | :-----: |:---------:|:---------:|:---------:|
|EPSANet-50(small) | 41.20 | 248.53 | 35.9 | 57.7 | 38.1 | 
|EPSANet-50(Large) | 46.50 | 271.10 | 37.1 | 59.0 | 39.5 | 

