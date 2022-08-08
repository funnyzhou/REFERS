# coding=utf-8
import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
import pdb

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import ml_collections

logger = logging.getLogger(__name__)

class fusion(nn.Module):
    def __init__(self, feature_size = 768): 
        super(fusion, self).__init__()
        self.fc1 = Linear(feature_size*3, 1) # 这里可以做两层得到1，也可以得到一个768维的weight
        self.fc2 = Linear(feature_size*3, 1)
        self.fc3 = Linear(feature_size*3, 1)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x1, x2, x3):
        # pdb.set_trace()
        batch_size = x1.size()[0]

        x1 = x1.view(-1, 768)
        x2 = x2.view(-1, 768)
        x3 = x3.view(-1, 768)

        x123 = torch.cat((x1, x2), 1)
        x123 = torch.cat((x123, x3), 1)

        weight1 = self.fc1(x123)
        weight2 = self.fc2(x123)
        weight3 = self.fc3(x123)

        weight1 = self.sigmoid(weight1)
        weight2 = self.sigmoid(weight2)
        weight3 = self.sigmoid(weight3)

        weight1 = weight1.view(batch_size, -1).unsqueeze(2)
        weight2 = weight2.view(batch_size, -1).unsqueeze(2)
        weight3 = weight3.view(batch_size, -1).unsqueeze(2)

        return weight1, weight2, weight3

