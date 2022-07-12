# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST
from torch.utils.data import Subset, Dataset

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def get_dataset(cls, cutout_length=0):
    

    if cls == "cifar10":
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
        transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]
        cutout = []
        if cutout_length > 0:
            cutout.append(Cutout(cutout_length))

        train_transform = transforms.Compose(transf + normalize + cutout)
        valid_transform = transforms.Compose(normalize)

        dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
    elif cls == "fmnist" or cls == 'fmnist2' or cls == 'fmnist3':
        MEAN = [0.28604063146254594]
        STD = [0.35302426207299326]
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.RandomHorizontalFlip()
        ]
        
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]
        train_transform = transforms.Compose(transf + normalize)
        valid_transform = transforms.Compose(normalize)

        
        dataset_train = FashionMNIST(root="../data", train=True, download=True, transform=train_transform)
        dataset_valid = FashionMNIST(root="../data", train=False, download=True, transform=valid_transform)
        
        if cls == 'fmnist2':
            class RemapSubset(Dataset):
                def __init__(self, subset, remap):
                    self.subset = subset
                    self.remap = remap 
                    
                def __len__(self):
                    return len(self.subset)
                
                def __getitem__(self, index):
                    x,y = self.subset[index]
                    return x, self.remap[y]
                
            targets = [0,1,3,4,5,7,8,9]
            remap = {0:0, 1:1, 3:2, 4:3, 5:4, 7:5, 8:6, 9:7}
            indices = [i for i, label in enumerate(dataset_train.targets) if label in targets]
            dataset_train = RemapSubset(Subset(dataset_train, indices), remap)
            
            indices = [i for i, label in enumerate(dataset_valid.targets) if label in targets]
            dataset_valid = RemapSubset(Subset(dataset_valid, indices), remap)
        if cls == 'fmnist3':
            class RemapSubset(Dataset):
                def __init__(self, subset, remap):
                    self.subset = subset
                    self.remap = remap 
                    
                def __len__(self):
                    return len(self.subset)
                
                def __getitem__(self, index):
                    x,y = self.subset[index]
                    return x, self.remap[y]
                
            targets = [2,6]
            remap = {2:0, 6:1}
            indices = [i for i, label in enumerate(dataset_train.targets) if label in targets]
            dataset_train = RemapSubset(Subset(dataset_train, indices), remap)
            
            indices = [i for i, label in enumerate(dataset_valid.targets) if label in targets]
            dataset_valid = RemapSubset(Subset(dataset_valid, indices), remap)
    else:   
        raise NotImplementedError
    return dataset_train, dataset_valid
