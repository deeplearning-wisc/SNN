from formatter import test
from pickle import FALSE
import torch
import os
import numpy as np
import torch.nn.functional as F
import time
import torchvision
import torchvision.transforms as transforms
import models.densenet as dn
import numpy as np
import time
import argparse
from feat_extract import feat_extract
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser(description='SNN testing with CIFAR benchmark')

parser.add_argument('--in-dataset', default="CIFAR-100", type=str, help='in-distribution dataset')
parser.add_argument('--model_arch', default='densenet', type=str, help='model architecture ([densenet, resnet50]')
parser.add_argument('--bs', default = 200, type = int, help='Batch size')

parser.set_defaults(argument=True)

args = parser.parse_args()
args.device = device


if __name__ == '__main__':
    feat_extract(args)
