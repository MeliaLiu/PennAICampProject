import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.io import read_image

def encode_label(label):
    labels = ['Snail', 'Water', 'Plant', 'Cattle', 'Grain', 'Bird', 'People', 'Animal', 'Snake']
    for i in range(len(labels)):
        if label == labels[i]:
            return i
    
    return -1

def decode_label(code):
    labels = ['Snail', 'Water', 'Plant', 'Cattle', 'Grain', 'Bird', 'People', 'Animal', 'Snake']
    return labels[code]


class HieroglyphicsDataset(Dataset):
    def __init__(self, directory_path, transform=None, target_transform=None):
        
        self.data = []
        self.transform = transform
        self.target_transform = target_transform
        
        print('You are creating a dataset from %s.' % (directory_path))
              
        for _, categories, _ in os.walk(directory_path):
            for category in categories:
                if (category.split('.')[-1] in ['DS_Store', 'ipynb_checkpoints']):
                    continue
                    
                label = encode_label(category)
                label = torch.tensor(label)
                if self.target_transform:
                    label = self.target_transform(label)
                    
                category_path = os.path.join(directory_path, category)
                for _, _, image_files in os.walk(category_path):
                    
                    # Add image 
                    count = 0
                    for image_file in image_files:
                        if not (image_file.split('.')[-1] in ['jpg', 'jpeg', 'png']):
                            continue
                        
                        image_path = os.path.join(category_path, image_file)
                        image = read_image(image_path)/255
                        # image = torch.from_numpy(image).double()
    
                        if self.transform:
                            image = self.transform(image)
                        self.data.append((image, label))
                        count += 1

                    # Summarize
                    print('\t%s - %d' % (category, count))
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    


class DemoDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.data = []
        self.transform = transform
        image = read_image(image_path)/255

        if self.transform:
            image = self.transform(image)
            
        self.data.append((image, torch.tensor(-1)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]