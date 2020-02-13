import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np
import math
from utils import *

from PIL import Image

class MattingDataSet(Dataset):
    def __init__(self, img_path, a_path, crop_size=(512, 512)):
        self.img_path = img_path
        self.a_path = a_path
        self.img_lst = os.listdir(self.img_path) 
        self.crop_size = crop_size        

    def __len__(self):

        return len(self.img_lst)

    def __getitem__(self, item):
        img_name, _ = self.img_lst[item].split(".")
        alpha = np.array(Image.open(os.path.join(self.a_path, img_name+'.png')))
        img = np.array(Image.open(os.path.join(self.img_path, img_name+'.jpg')))
        _kernel_size = np.random.randint(3,8,1).item()
        trimap = generate_trimap(alpha, (_kernel_size, _kernel_size))
        loc = random_sample(img, trimap, self.crop_size)
        img_patch = data_crop(img, loc, self.crop_size)
        trimap_patch = data_crop(trimap, loc, self.crop_size)
        alpha_patch = data_crop(alpha, loc, self.crop_size)
        unknown = binary_unknown(trimap_patch)[np.newaxis, :, :]
        img_patch = img_patch.transpose(2, 0, 1) / 255.0
        alpha_patch = alpha_patch[np.newaxis, :, :] / 255.0
        trimap_patch = trimap_patch[np.newaxis, :, :] / 255.0
        seg_patch = np.array(alpha_patch > 0.1, dtype=np.float)
        return {'img': img_patch,   'trimap': trimap_patch, 'alpha': alpha_patch, 'unknown':unknown, 'seg':seg_patch}


