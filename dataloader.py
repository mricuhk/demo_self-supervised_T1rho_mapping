import random

import pydicom
from pydicom.data import get_testdata_file
from pydicom import dcmread
import matplotlib.pyplot as plt
import scipy.io
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import nibabel as nib
import numpy as np
import cv2


def load_image (nifty_path):
    img = nib.load(nifty_path)

    images = img.get_fdata()
    slope = img.dataobj.slope
    inter = img.dataobj.inter
    scaled_image = slope * images + inter
    
    return scaled_image

class Demo_set(Dataset):
    def __init__(self,data_list):
        # input trianing_list is a list of slice file, which contains pairs
        num = len(data_list)
        self.data = data_list
        self.trans = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):

        input_dir = self.data[idx]
        input_image = load_image(input_dir)

        input_image = torch.tensor(input_image)

        return input_image


    def __len__(self):
        return len(self.data)


def load(dataset,batchsize):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, num_workers=1)

    return loader