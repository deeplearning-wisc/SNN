# How to Overcome Curse-of-Dimensionality for OOD Detection

## Preliminaries
It is tested under Ubuntu Linux 20.04 and Python 3.9 environment, and requries following packages to be installed:
* [PyTorch](https://pytorch.org/)
* [scipy](https://github.com/scipy/scipy)
* [numpy](http://www.numpy.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [faiss](https://github.com/facebookresearch/faiss)


## Usage

### 1. Dataset Preparation for CIFAR Experiment 

#### In-distribution dataset

The downloading process will start immediately upon running. 

#### Out-of-distribution dataset


We provide links and instructions to download each dataset:

* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `./ood_data/dtd`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `./ood_data/Places365`. We sample 10,000 images from the original test dataset. Download the sampled dataset from [here](https://drive.google.com/file/d/19MShiqHGdOZge0M9gQhwYfnZOJkVZ-mx/view?usp=share_link).
* [LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `./ood_data/LSUN`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `./ood_data/iSUN`.


[//]: # (For example, run the following commands in the **root** directory to download **LSUN**:)

[//]: # (```)

[//]: # (cd ./ood_data)

[//]: # (wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)

[//]: # (tar -xvzf LSUN.tar.gz)

[//]: # (```)

####  Pre-trained model

Please download [Pre-trained models](https://drive.google.com/file/d/1YP1oFzm6cFTR19ckzkky0Zixn3VBz47A/view?usp=share_link) and place in the `./checkpoints` folder for respective ID dataset and model architecture. For example: DenseNet-101 model trained on CIFAR-10 should be placed in `./checkpoints/CIFAR-10/densenet` folder.


#### Demo
##### 1. Demo code for training SNN on CIFAR benchmark

To train DenseNet-101 on CIFAR-100 dataset with subspace learning, run the following command:

```
python train_densenet.py --id CIFAR-100 --bs 64 --r 0.25
```
Run the following command, to train ResNet-50 on CIFAR-100 dataset:

```
python train_resnet.py --id CIFAR-100 --bs 128 --r 0.05
```


##### 2. Demo code for testing SNN on CIFAR benchmark

For inference, download the pre-trained models as mentioned above. To evaluate the OOD detection performance for a DenseNet model trained on CIFAR-100, run the following the command:

```
python test_cifar.py --in-dataset CIFAR-100 --model_arch densenet --bs 200
```
To run inference on a trained ResNet-50 model, run the following command:
```
python test_cifar.py --in-dataset CIFAR-100 --model_arch resnet50 --bs 200
```


### 2. Dataset Preparation for Large-scale Experiment (ImageNet-100)

#### In-distribution dataset

For generating the ImageNet-100 dataset, please follow the steps given [here](https://github.com/danielchyeh/ImageNet-100-Pytorch) and save the dataset in `./IN-100` folder.

#### Out-of-distribution dataset

We have 4 OOD datasets from 
[iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), 
[SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), 
[Places](http://places2.csail.mit.edu/PAMI_places.pdf), 
and [Textures](https://arxiv.org/pdf/1311.3618.pdf), 
and de-duplicated concepts overlapped with ImageNet-1k.

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset,
which can be download via the following links:
```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

For Textures, we use the entire dataset, which can be downloaded from their
[original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Please put all downloaded OOD datasets into `./ood_data`.

##### 1. Demo code for training SNN on Imagenet-100 benchmark

To train ResNet-101 on ImageNet-100 dataset with subspace learning, run the following command:
```
python train_imagenet.py --classes 100 --r 0.35
```
##### 2. Demo code for testing SNN on Imagenet-100 benchmark

The pretrained model can be downloaded from [here](https://drive.google.com/file/d/1YP1oFzm6cFTR19ckzkky0Zixn3VBz47A/view?usp=share_link). After downloading, place the pre-trained model in the `./checkpoints` folder. For inference on ImageNet-100, run the following command:
```
python feat_extract_largescale.py
```

## References
The codebase is adapted from [knn-ood](https://github.com/deeplearning-wisc/knn-ood).
