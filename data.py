import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, datasets, transforms
import os
from PIL import ImageFilter
import numpy as np
import random


root = './dataset'

class BaseDataset(Dataset):
    def __init__(self, config):
        self.config = config
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    

def get_dataset(config, mode='train', multiview=True):
    root = config.dataset.root
    name = config.dataset.name
    dataset_path = os.path.join(root, name, mode)
        
    if mode.startswith('linear'):
        dataset_path = os.path.join(root, name, 'train')
    
    if multiview:
        transform = MultiviewTransform(config, mode=mode)
    else:
        transform = base_augment(config, mode=mode)
    
    if name.startwith('ImageNet'):
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        
    if name.startswith('cifar'):
        dataset = datasets.CIFAR10(root=root, download=True, train=(mode != 'val'), transform=transform)
    
    return dataset

    
def get_loader(config, dataset, shuffle=True, sampler=None):
    if sampler:
        return DataLoader(dataset, batch_size=config.train.batch_size, drop_last=True, num_workers=config.system.num_workers, pin_memory=True, sampler=sampler)
    return DataLoader(dataset, batch_size=config.train.batch_size, shuffle=shuffle, num_workers=config.system.num_workers, drop_last=True)


class MultiviewTransform:
    def __init__(self, config, mode='train', num_view=2):
        self.config = config
        self.num_view = num_view
        self.transform = base_augment(config, mode=mode)
    
    def __call__(self, sample):
        views = []
        for _ in range(self.num_view):
            view = self.transform(sample)
            views.append(view)
        
        return views
    

def base_augment(config, mode='train'):
    img_size = config.dataset.img_size[0]
    
    if mode != 'val':
        crop = transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.))
    else:
        if config.dataset.name.startswith('cifar'):
            resize = 40
            orig_size = 32
        elif config.dataset.name.startswith('ImageNet'):
            resize = 256
            orig_size = 224
        crop = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(orig_size)
        ])
        
    flip = transforms.RandomHorizontalFlip()
    color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8) # simclr -> 0.4, 0.4, 0.4, 0.2
    gray_scale = transforms.RandomGrayscale(0.2)
    gaussian_blur = transforms.RandomApply([GaussianBlur()], p=0.5)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transforms_list = np.array([crop, color_jitter, gray_scale, gaussian_blur, flip, to_tensor, normalize])

    if mode == 'train':
        augment_mask = np.array([True, True, True, True, True, True, True])
    elif mode == 'linear':
        augment_mask = np.array([True, False, False, False, True, True, True])
    elif mode == 'val':
        augment_mask = np.array([True, False, False, False, False, True, True])
    else:
        raise NotImplementedError
    
    transform = transforms.Compose(transforms_list[augment_mask])
    
    return transform


class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
        
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        return x
        