# Axial-DeepLab (ECCV 2020, Spotlight)

This is an on-going PyTorch re-implementation of the [Axial-DeepLab paper](https://arxiv.org/abs/2003.07853):
```BibTeX
@inproceedings{wang2020axial,
  title={Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation},
  author={Wang, Huiyu and Zhu, Yukun and Green, Bradley and Adam, Hartwig and Yuille, Alan and Chen, Liang-Chieh},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```
This re-implementation is authored by an amazing junior student, [Huaijin Pi](https://huaijinpi.com/).

## Introduction

This repository is built for stand-alone axial-attention. 
Now we support the training and testing code of image 
classification on the imagenet dataset.

## How to use 

### Prepare datasets

```shell
mkdir data
cd data
ln -s path/to/dataset imagenet
```

### Training

- Non-distributed training

Specify gpus with gpu_id

```shell
python train.py --model axial50s --gpu_id 0,1,2,3 --batch_size 128 --val_batch_size 128 --name axial50s --lr 0.05 --nesterov
```

- Distributed training

```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python dist_train.py --model axial50s --batch_size 128 --val_batch_size 128 --name axial50s --lr 0.05 --nesterov --dist-url 'tcp://127.0.0.1:4128' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
```

You can change the model name to train different models.

### Testing

```shell
python train.py --model axial50s --gpu_id 0,1,2,3 --batch_size 128 --val_batch_size 128 --name axial50s --lr 0.05 --nesterov --test
```

You can test with distributed settings in the same way.
