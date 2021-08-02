#coding=utf-8

import sys
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Data(Dataset):
    def __init__(self, args):
        self.args      = args
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(685, 685),
            ToTensorV2()
        ])

        self.samples = []
        with open(args.datapath+'/'+args.list, 'r') as lines:
            for line in lines:
                name, label = line.strip().split(' ')
                self.samples.append([name, int(label)])

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        image       = cv2.imread(self.args.datapath+'/val/'+name)
        image       = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image       = self.transform(image=image)['image']
        return name, image, label

    def __len__(self):
        return len(self.samples)


class Test(object):
    def __init__(self, Data, args):
        ## dataset
        self.args    = args 
        self.data    = Data(args)
        self.loader  = DataLoader(self.data, batch_size=args.batch_size, pin_memory=False, shuffle=True, num_workers=args.num_workers)
        ## model
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        self.model.train(False)
        self.model.cuda()

    def cls_save(self):
        top1top5, cnt = {}, 0
        with torch.no_grad():
            for name, image, label in self.loader:
                image  = image.cuda().float()
                pred   = self.model(image).cpu()
                index  = torch.argsort(pred, dim=-1, descending=True)
                score  = (index==label.unsqueeze(1))
                for n, s in zip(name, score):
                    top1 = True if s[:1].sum().item()==1 else False
                    top5 = True if s[:5].sum().item()==1 else False
                    top1top5[n] = [top1, top5]
                cnt += len(name)
                print(cnt)
            np.save('top1top5', top1top5)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath'    ,default='../dataset/ImageNet2012')
    parser.add_argument('--mode'        ,default='val')
    parser.add_argument('--list'        ,default='val.txt')
    parser.add_argument('--batch_size'  ,default=64)
    parser.add_argument('--num_workers' ,default=8)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    t = Test(Data, args)
    t.cls_save()