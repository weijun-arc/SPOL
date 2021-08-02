#coding=utf-8

import sys
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
plt.ion()
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import Model
from utils import IoU, gaussian
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Data(Dataset):
    def __init__(self, args):
        self.args      = args
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc'))

        self.samples  = []
        self.cls2idx  = np.load(args.datapath+'/cls2idx.npy', allow_pickle=True).item()
        self.top1top5 = np.load(args.top1top5, allow_pickle=True).item()
        with open(args.datapath+'/'+args.list, 'r') as lines:
            for line in lines:
                name, label = line.strip().split(' ')
                if name not in ['ILSVRC2012_val_00033202.JPEG']: # wrong labels
                    self.samples.append([name, int(label)])

    def __getitem__(self, idx):
        name, label = self.samples[idx]
        image  = cv2.imread(self.args.datapath+'/val/'+name)
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tree   = ET.parse(self.args.datapath+'/box/'+name.replace('.JPEG', '.xml'))
        H,W,C  = image.shape
        bboxes = []
        for obj in tree.findall('object'):
            cls = self.cls2idx[obj.find('name').text]
            if cls != label:
                continue
            box = obj.find('bndbox')
            xmin, ymin, xmax, ymax = float(box.find('xmin').text), float(box.find('ymin').text), float(box.find('xmax').text), float(box.find('ymax').text)
            bboxes.append([min(max(xmin, 0), W-1), min(max(ymin, 0), H-1), max(min(xmax, W-1), 0), max(min(ymax, H-1), 0), label])
            pairs = self.transform(image=image, bboxes=bboxes)
        return name, image, pairs['image'], pairs['bboxes'], self.top1top5[name]

    def __len__(self):
        return len(self.samples)


class Test(object):
    def __init__(self, Data, Model, args):
        self.args   = args
        ## dataset
        self.data   = Data(args)
        self.loader = DataLoader(self.data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        ## network
        self.model  = Model(args)
        self.model.train(False)
        self.model.cuda()
    
    def box_precise(self):
        with torch.no_grad():
            cnt, cnt_top1, cnt_top5, cnt_loc, thresh = 0, 0, 0, 0, 0.8
            for name, origin, image, bboxes, top1top5 in self.loader:
                image = image.cuda().float()
                top1, top5 = top1top5
                if len(bboxes)==0:
                    print(name)
                    continue
                label  = bboxes[0][-1].item()
                bboxes = [[box[0].item(), box[1].item(), box[2].item(), box[3].item()] for box in bboxes]
                pred   = self.model(image)
                pred   = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=True)
                pred   = torch.sigmoid(pred)[0, 0, :, :].cpu().numpy()

                ## accuracy
                mask     = np.uint8(pred>thresh)*255
                contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                if len(contours)==0:
                    continue
                contour  = max(contours, key=cv2.contourArea)
                x,y,w,h  = cv2.boundingRect(contour)
                if max([IoU([x, y, x+w, y+h],  box) for box in np.array(bboxes)])>0.5:
                    cnt_loc  += 1
                    cnt_top1 += 1 if top1 else 0
                    cnt_top5 += 1 if top5 else 0
                cnt += 1
                if cnt%100 == 0:
                    print('count=%d | top1=%.5f | top5=%.5f | GT-Known=%.5f'%(cnt, cnt_top1/cnt, cnt_top5/cnt, cnt_loc/cnt))
            print('count=%d | top1=%.5f | top5=%.5f | GT-Known=%.5f'%(cnt, cnt_top1/cnt, cnt_top5/cnt, cnt_loc/cnt))

            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath'    ,default='../dataset/ImageNet2012')
    parser.add_argument('--top1top5'    ,default='../imagenet-cls/top1top5.npy')
    parser.add_argument('--snapshot'    ,default='./out/model-6000')
    parser.add_argument('--mode'        ,default='val')
    parser.add_argument('--list'        ,default='val.txt')
    parser.add_argument('--batch_size'  ,default=1)
    parser.add_argument('--num_workers' ,default=1)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    t = Test(Data, Model, args)
    t.box_precise()