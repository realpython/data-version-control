from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


#FOLDERS_TO_LABELS = {"n03445777": "golf ball", "n03888257": "parachute"}

def get_train_valid_loader(data_dir_train,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    
    # define transforms
    valid_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
        ])


    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir_train, train=True,
        download=True, transform=train_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    return train_loader

def get_test_loader(data_dir_test,
                    batch_size,
                    shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir_test, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader


# CIFAR10 dataset 
train_loaderr = get_train_valid_loader(data_dir_train = './data/raw/train',                                      batch_size = 64,
                       augment = False,                             		     random_seed = 1)

test_loader = get_test_loader(data_dir_test = './data/raw/test',
                              batch_size = 64)




def main(repo_path):
    data_path = repo_path / "data"
    data_dir_train = data_path / "raw/train"
    data_dir_test = data_path / "raw/test"
    train_path = data_path / "raw/train"
    test_path = data_path / "raw/val"
    train_loader = get_train_valid_loader(data_dir_train = './data/raw/train',                                      batch_size = 64,
                        augment = False,                             		     random_seed = 1)

    test_loader = get_test_loader(data_dir_test = './data/raw/test',
                                batch_size = 64)


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
