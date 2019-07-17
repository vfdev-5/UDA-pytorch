import numpy as np

from torch.utils.data import Subset, Dataset, DataLoader

from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip

from cifar.datasets import get_train_test_datasets as get_cifar_train_test_datasets
from cifar import fastresnet

from utils import autoaugment
from utils.transforms import RandomErasing


def set_seed(seed):

    import random
    import numpy as np
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_train_test_loaders(dataset_name, path, num_labelled_samples, batch_size, num_workers, unlabelled_batch_size=None, pin_memory=True):

    if "cifar" in dataset_name.lower():
        train_ds, test_ds, num_classes = get_cifar_train_test_datasets(dataset_name, path)
    else:
        raise RuntimeError("Unknown dataset '{}'".format(dataset_name))

    train_labelled_ds, train_unlabelled_ds = \
        stratified_train_labelled_unlabelled_split(train_ds, 
                                                   num_labelled_samples=num_labelled_samples, 
                                                   num_classes=num_classes, seed=12)

    if "cifar" in dataset_name.lower():
        train_transform = Compose([
            Pad(4),
            RandomCrop(32, fill=128),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            RandomErasing(scale=(0.1, 0.33)),
        ])

        test_transform = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        unsupervised_train_transformation = Compose([
            Pad(4),
            RandomCrop(32, fill=128),
            autoaugment.CIFAR10Policy(),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            RandomErasing(scale=(0.1, 0.33)),
        ])

        train_labelled_ds = TransformedDataset(train_labelled_ds, 
                                               transform_fn=lambda dp: (train_transform(dp[0]), dp[1]))
        test_ds = TransformedDataset(test_ds, transform_fn=lambda dp: (test_transform(dp[0]), dp[1]))

        original_transform = lambda dp: train_transform(dp[0])
        augmentation_transform = lambda dp: unsupervised_train_transformation(dp[0])
        train_unlabelled_ds = TransformedDataset(train_unlabelled_ds, UDATransform(original_transform, augmentation_transform))

    if unlabelled_batch_size is None:
        unlabelled_batch_size = batch_size

    train_labelled_loader = DataLoader(train_labelled_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_unlabelled_loader = DataLoader(train_unlabelled_ds, batch_size=unlabelled_batch_size, num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, num_workers=num_workers, pin_memory=pin_memory)

    return train_labelled_loader, train_unlabelled_loader, test_loader


def get_model(name):
    
    fn = None
    if name in models.__dict__:
        fn = models.__dict__[name]
    elif name in fastresnet.__dict__:
        fn = fastresnet.__dict__[name]
    else:
        raise RuntimeError("Unknown model name {}".format(name))

    return fn()


class TransformedDataset(Dataset):

    def __init__(self, ds, transform_fn):
        assert isinstance(ds, Dataset)
        assert callable(transform_fn)
        self.ds = ds
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        dp = self.ds[index]
        return self.transform_fn(dp)


class UDATransform:

    def __init__(self, original_transform, augmentation_transform, copy=False):
        self.original_transform = original_transform
        self.augmentation_transform = augmentation_transform
        self.copy = copy

    def __call__(self, dp):        
        if self.copy:
            aug_dp = dp.copy()
        else:
            aug_dp = dp
        tdp1 = self.original_transform(dp)
        tdp2 = self.augmentation_transform(aug_dp)
        return tdp1, tdp2 


def stratified_train_labelled_unlabelled_split(ds, num_labelled_samples, num_classes, seed=None):
    labelled_indices = []
    unlabelled_indices = []
    
    if seed is not None:
        np.random.seed(seed)

    indices = np.random.permutation(len(ds))
    
    class_counters = list([0] * num_classes)
    max_counter = num_labelled_samples // num_classes
    for i in indices:
        dp = ds[i]        
        
        if num_labelled_samples < sum(class_counters):
            unlabelled_indices.append(i)
        else:
            y = dp[1]        
            c = class_counters[y]
            if c < max_counter:
                class_counters[y] += 1
                labelled_indices.append(i)
            else:
                unlabelled_indices.append(i)

    assert len(set(labelled_indices) & set(unlabelled_indices)) == 0, \
        "{}".format(set(labelled_indices) & set(unlabelled_indices))
    
    train_labelled_ds = Subset(ds, labelled_indices)
    train_unlabelled_ds = Subset(ds, unlabelled_indices)
    return train_labelled_ds, train_unlabelled_ds
