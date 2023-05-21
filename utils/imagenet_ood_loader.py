import os
import torch
import torchvision
from torchvision import transforms
from easydict import EasyDict




transform_train_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

kwargs = {'num_workers': 2, 'pin_memory': True}

def get_loader_in(batch_size, config_type='default'):
    config = EasyDict({
        "default": {
            'batch_size': batch_size,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
        },
        "eval": {
            'batch_size': batch_size,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_test_largescale,
        },
    })[config_type]
    root = './IN-100'
    train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'train'), config.transform_train_largescale),
                batch_size=config.batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=False, **kwargs)

    return EasyDict({
        "train_loader": train_loader,
        "val_loader": val_loader,
    })


def get_loader_out(batch_size, val_dataset, config_type='default'):

    config = EasyDict({
        "default": {
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
            'batch_size': batch_size
        },
    })[config_type]
    imagesize = 224
    if val_dataset == 'dtd':
        transform = config.transform_test_largescale 
        val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root="./ood_data/dtd/images", transform=transform),
                                                    batch_size=batch_size, shuffle=True, num_workers=2)
    
    elif val_dataset == 'places50':
        val_ood_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder("./ood_data/Places",
                                                transform=config.transform_test_largescale), batch_size=batch_size,
            shuffle=False, num_workers=2)
    elif val_dataset == 'sun50':
        val_ood_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder("./ood_data/SUN",
                                                transform=config.transform_test_largescale), batch_size=batch_size,
            shuffle=False,
            num_workers=2)
    elif val_dataset == 'inat':
        val_ood_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder("./ood_data/iNaturalist",
                                                transform=config.transform_test_largescale), batch_size=batch_size,
            shuffle=False,
            num_workers=2)
   
    elif val_dataset == 'imagenet':
        val_ood_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(os.path.join('./IN-100', 'val'), config.transform_test_largescale),
            batch_size=config.batch_size, shuffle=True, **kwargs)

    return val_ood_loader