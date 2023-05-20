
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
import models.resnet as resnet
import numpy as np
import time
from run_snn import run_knn_func
from pathlib import Path
from types import MethodType

def check_valid(path):
    path = Path(path)
    return not path.stem.startswith('._')

def id_loader(args):
    in_dataset = args.in_dataset
    bs = args.bs

    if in_dataset == "CIFAR-10":
            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    elif in_dataset == "CIFAR-100":
            normalize = transforms.Normalize(mean = [0.507,0.487,0.441], std = [0.267, 0.256, 0.276])
    
    transform_test = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                normalize,
            ])

    if in_dataset == "CIFAR-10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True, num_workers=2)
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
        trainloaderIn = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
        num_classes = 10

    elif in_dataset == "CIFAR-100":
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True, num_workers=2)
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_test)
        trainloaderIn = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
        num_classes = 100
    
    args.num_classes = num_classes
    args.transform_test = transform_test
    return testloaderIn, trainloaderIn, args


def model_loader(args):
    model_arch = args.model_arch
    num_classes = args.num_classes
    if model_arch == 'densenet':
        print("Densenet")
        model = dn.DenseNet3(100, num_classes, growth_rate= 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=None)
        checkpoint = torch.load(
            "./checkpoints/{in_dataset}/densenet/model_best.pth.tar".format(in_dataset=args.in_dataset))
        model.load_state_dict(checkpoint['state_dict'], strict = True)

    elif model_arch == 'resnet50':
        model = resnet.ResNet50(num_class=num_classes)
        checkpoint = torch.load(
                "./checkpoints/{in_dataset}/resnet50/model_best.pth.tar".format(in_dataset=args.in_dataset))
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['net'].items()}
        model.load_state_dict(state_dict)

    else:
        assert False, 'Not supported model arch: {}'.format(model_arch)
    
    model.cuda()
    model.eval()
    return model



def get_out_loader(out_dataset, args):
    batch_size = args.bs
    transform = args.transform_test
    if out_dataset == 'SVHN':
        testsetout = torchvision.datasets.SVHN('~/Documents/ood_data/', split='test', transform=transform, download=True)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root="~/Documents/ood_data/dtd/images", transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    elif out_dataset == 'places365':
        testsetout = torchvision.datasets.ImageFolder(root="~/Documents/ood_data/Places365/", transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    elif out_dataset == 'CIFAR-10':
        testsetout = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    elif out_dataset == 'CIFAR-100':
        testsetout = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        testsetout = torchvision.datasets.ImageFolder("~/Documents/ood_data/{}".format(out_dataset), transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
    return testloaderOut

def feat_extract(args):
    
    FORCE_RUN = True
    testloaderIn , trainloaderIn, args = id_loader(args)
    
    print(f"{args.in_dataset} with {args.num_classes} classes")
    model = model_loader(args)

    dummy_input = torch.zeros((1, 3, 32, 32)).cuda()
    score, feature_list = model.feature_list(dummy_input)
    featdims = [feat.shape[1] for feat in feature_list]
    start = sum(featdims[:-1])
    end = start + featdims[-1]
    print(featdims)
    print(start, end)
    begin = time.time()
    num_classes = args.num_classes; batch_size = args.bs
    for split, in_loader in [('train', trainloaderIn), ('val', testloaderIn),]:
    
        cache_name = f"cache/{args.in_dataset}_{args.model_arch}_{split}_in_alllayers.npy"
        if FORCE_RUN or not os.path.exists(cache_name):

            feat_log = np.zeros((len(in_loader.dataset), sum(featdims)))

            score_log = np.zeros((len(in_loader.dataset), num_classes))
            label_log = np.zeros(len(in_loader.dataset))

            model.eval()
            for batch_idx, (inputs, targets) in enumerate(in_loader):
        
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))

                score, feature_list = model.feature_list(inputs)
            
                out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)
            
                feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                label_log[start_ind:end_ind] = targets.data.cpu().numpy()
                score_log[start_ind:end_ind] = score.data.cpu().numpy()
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(in_loader)}")
            np.save(cache_name, (feat_log.T, score_log.T, label_log))
        else:
            feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
            feat_log, score_log = feat_log.T, score_log.T
    
    d = ['SVHN', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365'] 
    for ood_dataset in d:
        
        out_loader = get_out_loader(ood_dataset, args)
        cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.model_arch}_out_alllayers.npy"
        if FORCE_RUN or not os.path.exists(cache_name):
            ood_feat_log = np.zeros((len(out_loader.dataset), sum(featdims)))
            ood_score_log = np.zeros((len(out_loader.dataset), num_classes))

            model.eval()
            for batch_idx, (inputs, _) in enumerate(out_loader):
                inputs = inputs.to(args.device)
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

                score, feature_list = model.feature_list(inputs)
                out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)

                ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                ood_score_log[start_ind:end_ind] = score.data.cpu().numpy()
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(out_loader)}")
            np.save(cache_name, (ood_feat_log.T, ood_score_log.T))
        else:
            ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
            ood_feat_log, ood_score_log = ood_feat_log.T, ood_score_log.T
    print(time.time() - begin)
    run_knn_func(args.in_dataset, args.model_arch, d, start, end)
   