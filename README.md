# Point-cloud-down-sampling-strategies-for-deep-learning-based-crop-organ-segmentation
This repo contains the official codes for our paper:

### **A comparative study on point cloud down-sampling strategies for deep learning-based crop organ segmentation**

[Dawei Li](https://davidleepp.github.io/),  [Yongchang Wei](https://github.com/WeiyongchangChina), and Rongsheng Zhu

Published on *Plant Methods* in 2023

[[Paper](https://link.springer.com/article/10.1186/s13007-023-01099-7)]

___

## Prerequisites

- Python == 3.7.13
- Numpy == 1.21.5
- tensorflow == 1.13.1
- CUDA == 11.7
- cuDNN == 10.1

## Abstract

The 3D crop data obtained during cultivation is of great signifcance to screening excellent varieties in modern breeding and improvement on crop yield. With the rapid development of deep learning, researchers have been making  innovations in aspects of both data preparation and deep network design for segmenting plant organs from 3D  data. 

Training of the deep learning network requires the input point cloud to have a fxed scale, which means all  point clouds in the batch should have similar scale and contain the same number of points. A good down-sampling  strategy can reduce the impact of noise and meanwhile preserve the most important 3D spatial structures. As far  as we know, this work is the frst comprehensive study of the relationship between multiple down-sampling strategies and the performances of popular networks for plant point clouds. 

Five down-sampling strategies (including  FPS, RS, UVS, VFPS, and 3DEPS) are cross evaluated on fve diferent segmentation networks (including PointNet+ +,  DGCNN, PlantNet, ASIS, and PSegNet). The overall experimental results show that currently there is no strict golden  rule on fxing down-sampling strategy for a specifc mainstream crop deep learning network, and the optimal downsampling strategy may vary on diferent networks. However, some general experience for choosing an appropriate  sampling method for a specifc network can still be summarized from the qualitative and quantitative experiments. 

 First, 3DEPS and UVS are easy to generate better results on semantic segmentation networks. Second, the voxel-based  down-sampling strategies may be more suitable for complex dual-function networks. Third, at 4096-point resolution,  3DEPS usually has only a small margin compared with the best down-sampling strategy at most cases, which means  3DEPS may be the most stable strategy across all compared. This study not only helps to further improve the accuracy  of point cloud deep learning networks for crop organ segmentation, but also gives clue to the alignment of downsampling strategies and a specifc network

![](docs/down-sampling&network.png)

## 1.File Structure

Abstract  
├─data-example  
│  ├─3DEPS_ratio=0.20  
│  └─raw-data  
├─deep-learning-network  
│  ├─ASIS  
│  │  ├─data  
│  │  ├─models  
│  │  ├─log  
│  │  │  ├─test  
│  │  │  └─train  
│  │  ├─tf_ops  
│  │  │  ├─3d_interpolation  
│  │  │  ├─grouping  
│  │  │  └─sampling  
│  │  └─utils  
│  ├─DGCNN  
│  │  ├─data  
│  │  ├─models  
│  │  ├─part_seg  
│  │  │  ├─log  
│  │  │  │  ├─test  
│  │  │  │  └─train  
│  │  ├─tf_ops  
│  │  │  ├─3d_interpolation  
│  │  │  ├─grouping  
│  │  │  │  ├─test  
│  │  │  └─sampling  
│  │  └─utils  
│  ├─PlantNet  
│  │  ├─data  
│  │  ├─models  
│  │  │  ├─log_test  
│  │  │  │  ├─test  
│  │  │  │  └─train  
│  │  ├─tf_ops  
│  │  │  ├─3d_interpolation  
│  │  │  ├─grouping  
│  │  │  │  ├─test  
│  │  │  └─sampling  
│  │  └─utils  
│  ├─PointNet++  
│  │  ├─data  
│  │  ├─models  
│  │  ├─part_seg  
│  │  ├─tf_ops  
│  │  │  ├─3d_interpolation  
│  │  │  ├─grouping  
│  │  │  │  ├─test  
│  │  │  └─sampling  
│  │  └─utils  
│  └─PSegNet  
│      ├─data  
│      ├─models  
│      │  ├─log_test  
│      │  │  ├─test  
│      │  │  └─train  
│      ├─tf_ops  
│      │  ├─3d_interpolation  
│      │  ├─grouping  
│      │  │  ├─test  
│      │  └─sampling  
│      └─utils  
├─docs  
└─down-sampling-strategies  
    ├─3DEPS  
    ├─dataset-creation-process  
    ├─FPS  
    ├─RS  
    ├─UVS  
    └─VFPS  
