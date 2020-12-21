import torch
import torchvision
from transform import *


class DatasetCIFAR100(object):
    def __init__(self, train):
        self.transform = transform()
        if train:
            self.imgs = torchvision.datasets.CIFAR100(root='../data', train=True, transform=self.transform, download=True)
        else:
            self.imgs = torchvision.datasets.CIFAR100(root='../data', train=False, transform=self.transform)

    def __getitem__(self):
        return self.imgs

    def __len__(self):
        return len(self.imgs)
