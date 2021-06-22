# Axial-DeepLab (ECCV 2020, Spotlight)

News: The [official TF2 re-implementation](https://github.com/google-research/deeplab2/blob/main/g3doc/projects/axial_deeplab.md) is available in [DeepLab2](https://github.com/google-research/deeplab2). Axial-SWideRNet achieves 68.0% PQ or 83.5% mIoU on Cityscaspes validation set, with only *single-scale* inference and *ImageNet-1K* pretrained checkpoints.

This is a PyTorch re-implementation of the [Axial-DeepLab paper](https://arxiv.org/abs/2003.07853). The re-implementation is mainly done by an amazing senior student, [Huaijin Pi](https://huaijinpi.com/).
```BibTeX
@inproceedings{wang2020axial,
  title={Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation},
  author={Wang, Huiyu and Zhu, Yukun and Green, Bradley and Adam, Hartwig and Yuille, Alan and Chen, Liang-Chieh},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```
Currently, only ImageNet classification with the "Conv-Stem + Axial-Attention" backbone is supported. If you are interested in contributing to this repo, please open an issue and we can further discuss.


### Preparation

```shell
pip install tensorboardX
mkdir data
cd data
ln -s path/to/dataset imagenet
```

### Training

- Non-distributed training

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

### Model Zoo

| Method | Params (M) | Top-1 Acc (%) |
|:------:|:----------:|:-------------:|
|ResNet-26| 13.7 | 74.5 |
|[Axial-ResNet-26-S](http://www.cs.jhu.edu/~hwang157/axial26s.pth)|**5.9**|**75.8**|

## Credits

- ImageNet training script is modified from https://github.com/mit-han-lab/proxylessnas

- ImageNet distributed training script is modified from https://github.com/pytorch/examples/tree/master/imagenet

- ResNet is modified from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
