import time
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.utils import create_2d_circular_mask, image_patching, normalize_array
from utils.normalizations import relion_normalization, denoise_jpg_image_reduced_torch

class Train_Dataset(Dataset):
    '''
    Dataset class that transforms training coordinates for the 3D Unet

    Parameters
    ----------
    input_dataset_path : str
        Path of validation data

    train_data_list : list
        List of images names

    Attributes
    ----------
    dataset : list
        List that contains dataset information in coo matrix format.

    n_samples : int
        Length of dataset instances.
    '''
    def __init__(self, config):
        self.x, self.y = [], []
        self.mean, self.std = [], []
        a = time.time()
        dataset_len = len(config.train_data_list)
        for i, image_id in enumerate(config.train_data_list):
            x_temp = np.array(Image.open(f'{config.input_dataset_path}{image_id}'))/255
            if not config.random_resized_crop:
                #Split Image into patches
                x_temp = image_patching(x_temp,config.initial_img_patches_num, config.img_size)[0,0]
                self.x += list(x_temp)
            else:
                self.x.append(x_temp)
            #Load image-target and append
            print(f'Training dataset creation step {i+1}/{dataset_len}.',end='\r')
        print(f'Training dataset was created in {round(time.time()-a,2)} seconds.')
        self.n_samples = len(self.x)
        self.x = np.array(self.x)
        if config.standarization:
            config.mean, config.std = np.mean(self.x), np.std(self.x)
            self.x = (self.x-config.mean)/config.std
    def __getitem__(self,index):
        return self.x[index]

    def __len__(self):
        #Length of your dataset
        return self.n_samples

class Validation_Dataset(Dataset):
    '''
    Dataset class that transforms validation coordinates for the 3D Unet

    Parameters
    ----------
    validation_data_path : str
        Path of validation data

    validation_data_list : list
        List of images names

    Attributes
    ----------
    dataset : list
        List that contains dataset information in coo matrix format.

    n_samples : int
        Length of dataset instances.
    '''
    def __init__(self, config):
        self.x, self.y = [], []
        a = time.time()
        dataset_len = len(config.validation_data_list)
        for i, image_id in enumerate(config.validation_data_list):
            #Load input map
            x_temp = np.array(Image.open(f'{config.input_dataset_path}{image_id}'))/255
            #Split Image into patches
            x_temp = image_patching(x_temp,config.initial_img_patches_num, config.img_size)[0,0]
            self.x += list(x_temp)
            print(f'Validation dataset creation step {i+1}/{dataset_len}.',end='\r')
        print(f'Validation dataset was created in {round(time.time()-a,2)} seconds.')
        self.n_samples = len(self.x)
        self.x = np.array(self.x)
        if config.standarization:
            self.x = (self.x-config.mean)/config.std
    def __getitem__(self,index):
        return self.x[index]

    def __len__(self):
        #Length of your dataset
        return self.n_samples
