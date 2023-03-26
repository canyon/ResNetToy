import os
import torch
import numpy as np
from utils.cifardataset import CIFAR
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utils.cutout import Cutout

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Number of images loaded per batch
batch_size = 16
# number of subprocesses to use for data loading
nw = os.cpu_count()
# nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
print('Using {} dataloader workers every process'.format(nw))
num_workers = nw
# percentage of training set to use as validation
valid_size = 0.1

def read_dataset(batch_size=16,valid_size=0.2,num_workers=0):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    transform_train = transforms.Compose([
        # Fill with 0 around first, then randomly crop the image into 32*32
        transforms.RandomCrop(32, padding=4), 
        # The image has half the probability of flipping and half of the probability of not flipping
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # The mean and variance used in the normalization of each layer of R, G, and B
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), 
        Cutout(n_holes=1, length=8),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    
    ##########################################== Edit Below ==#######################################
    # Convert data to torch.FloatTensor, and normalize
    dataset_base_path = 'C:/Users/79981/Desktop/CAS771/code/ResNetToy/dataset/10/'
    #################################################################################################

    npy_path_train=os.path.join(dataset_base_path, 'train')
    txt_path_train=os.path.join(dataset_base_path, 'train_label.txt')
    npy_path_test=os.path.join(dataset_base_path, 'test')
    txt_path_test=os.path.join(dataset_base_path, 'test_label.txt')
    train_data = CIFAR(npy_path=npy_path_train, txt_path=txt_path_train, transform=transform_train)
    valid_data = CIFAR(npy_path=npy_path_train, txt_path=txt_path_train, transform=transform_test)
    test_data = CIFAR(npy_path=npy_path_test, txt_path=txt_path_test, transform=transform_test)
        

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    # Samples sample elements at the given list of indices without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader,valid_loader,test_loader