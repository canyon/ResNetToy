import os
import torch
import numpy as np
from PIL import Image
# from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms

class CIFAR(VisionDataset):
    def __init__(self, npy_path, txt_path, transform):
        self.npy_path = npy_path
        self.txt_path = txt_path
        self.transform = transform
        
        # Load labels into a dictionary
        self.labels = {}
        with open(txt_path, 'r') as f:
            for line in f:
                filename, class_num = line.strip().split()
                self.labels[filename] = int(class_num)

        # Load image filenames into a list
        self.filenames = list(self.labels.keys())
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        # Load image and label
        filename = self.filenames[index]
        img = np.load(os.path.join(self.npy_path, filename))
        img = Image.fromarray(np.uint8(img))
        # img.show()
        label = self.labels[filename]
        
        # Apply transforms
        img = self.transform(img)
        
        return img, label
