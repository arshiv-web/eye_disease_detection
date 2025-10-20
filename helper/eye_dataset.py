import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import glob
import pickle

from PIL import Image

import numpy as np
import pandas as pd

from helper.image_resizer import mean_std, Resize


class EyeDataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, id):
        image = Image.open(self.images[id])
        label = self.labels[id]

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label
    

def load(data_path, data_type='train', extns=['jpeg', 'jpg', 'png','tiff', 'tif', 'gif', 'bmp']):
    image_paths = []
    labels = []

    class_folders = [dir for dir in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, dir))]
    class2label = {class_name: label for label, class_name in enumerate(class_folders)}

    for class_name in class_folders:
        full_path = os.path.join(data_path, class_name)

        for extn in extns:
            image_paths.extend(glob.glob(os.path.join(full_path, f'*.{extn}')))
            labels.extend(len(glob.glob(os.path.join(full_path, f'*.{extn}'))) * [class2label[class_name]])
    
    if data_type=='train':
        data_transforms = transforms.Compose([Resize(256,256), transforms.ToTensor()])
        dataset = EyeDataset(image_paths, labels, data_transforms)

        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        mean, std = mean_std(data_loader)
        with open('eye_mean_std.pk1', 'wb') as f:
            pickle.dump((mean, std), f)
        with open('class2label.pk1', 'wb') as f:
            pickle.dump(class2label, f)
    
    else:
        with open('eye_mean_std.pk1', 'rb') as f:
            mean, std = pickle.load(f)
        
    if data_type == 'train':
        data_transforms = transforms.Compose([
            Resize(256,256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        data_transforms = transforms.Compose([
            Resize(256,256),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    dataset = EyeDataset(image_paths, labels, data_transforms)
    return dataset

def loadSingle(image_path):
    image_paths = [image_path]
    labels = [0] #doesnt really matter
    with open('eye_mean_std.pk1', 'rb') as f:
        mean, std = pickle.load(f)
    data_transforms = transforms.Compose([
        Resize(256,256),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset = EyeDataset(image_paths, labels, data_transforms)
    return dataset

if __name__ == '__main__':
    load('splitDataset/study')