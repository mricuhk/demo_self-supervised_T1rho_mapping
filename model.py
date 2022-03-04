import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import UNet

import random




class rho_CNN(nn.Module):

    def __init__(self, pretrain=1):
        super().__init__()
        self.unet = UNet()

    def forward(self, x):
        x = self.unet(x)

        return x

