#coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import ResNet50
from utils import weight_init

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args    = args
        self.bkbone  = ResNet50()
        self.fc51    = nn.Conv2d(2048,  128, kernel_size=1, stride=1, padding=0)
        self.bn51    = nn.BatchNorm2d(128)
        self.fc52    = nn.Conv2d( 128, 2048, kernel_size=1, stride=1, padding=0)
        self.fc41    = nn.Conv2d(1024,  128, kernel_size=1, stride=1, padding=0)
        self.bn41    = nn.BatchNorm2d(128)
        self.fc42    = nn.Conv2d( 128, 1024, kernel_size=1, stride=1, padding=0)
        self.fc31    = nn.Conv2d( 512,  128, kernel_size=1, stride=1, padding=0)
        self.bn31    = nn.BatchNorm2d(128)
        self.fc32    = nn.Conv2d( 128,  512, kernel_size=1, stride=1, padding=0)

        self.linear5 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0)
        self.bn5     = nn.BatchNorm2d(128)
        self.linear4 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)
        self.bn4     = nn.BatchNorm2d(128)
        self.linear3 = nn.Conv2d( 512, 128, kernel_size=1, stride=1, padding=0)
        self.bn3     = nn.BatchNorm2d(128)
        self.linear  = nn.Conv2d(128, self.args.clsnum, kernel_size=1, stride=1, padding=0)
        self.initialize()

    def forward(self, x, shape=None):
        out2, out3, out4, out5 = self.bkbone(x)

        out5a  = F.relu(self.bn51(self.fc51(out5.mean(dim=(2,3), keepdim=True))), inplace=True)
        out4a  = F.relu(self.bn41(self.fc41(out4.mean(dim=(2,3), keepdim=True))), inplace=True)
        out3a  = F.relu(self.bn31(self.fc31(out3.mean(dim=(2,3), keepdim=True))), inplace=True)
        vector = out5a*out4a*out3a
        out5   = torch.sigmoid(self.fc52(vector))*out5
        out4   = torch.sigmoid(self.fc42(vector))*out4
        out3   = torch.sigmoid(self.fc32(vector))*out3
        
        out5 = F.relu(self.bn5(self.linear5(out5)), inplace=True)
        out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out4 = F.relu(self.bn4(self.linear4(out4)), inplace=True)
        out4 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out3 = F.relu(self.bn3(self.linear3(out3)), inplace=True)

        pred = out5*out4*out3
        if self.args.mode=='train':
            pred = F.dropout(pred, p=0.5)
        pred = F.relu(self.linear(pred), inplace=True)
        return pred

    def initialize(self):
        if self.args.snapshot:
            print('load model...')
            self.load_state_dict(torch.load(self.args.snapshot))
        else:
            weight_init(self)