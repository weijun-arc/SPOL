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
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc'))

        self.samples = []
        with open(args.datapath+'/'+args.list, 'r') as lines:
            for line in lines:
                name, label, box       = line.strip().split(',')
                xmin, ymin, xmax, ymax = box.split(' ')
                self.samples.append([name, [[float(xmin), float(ymin), float(xmax), float(ymax), int(label)]]])

    def __getitem__(self, idx):
        name, bboxes = self.samples[idx]
        image = cv2.imread(self.args.datapath+'/image/'+name)
        H,W,_ = image.shape
        bboxes_clip = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, label = bbox
            bboxes_clip.append([max(xmin, 0), max(ymin, 0), min(xmax, W-1), min(ymax, H-1), label])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pairs = self.transform(image=image, bboxes=bboxes_clip)
        return name, image, pairs['image'], pairs['bboxes']

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
    

    def pred_save(self):
        boxes = {}
        with torch.no_grad():
            for iter, (name, origin, image, bboxes) in enumerate(self.loader):
                name, origin, image, label = name[0], np.uint8(origin[0]), image.cuda().float(), bboxes[0][4].item()
                H,W,C       = origin.shape
                pred        = self.model(image)[0, label, :, :].cpu().numpy()
                pred        = pred/(pred.max()+1e-6)
                pred        = cv2.resize(pred, (256, 256), interpolation=cv2.INTER_LINEAR)

                ## gaussian
                weights     = pred.copy()
                weights[np.where(pred<0.1)] = 0
                weights[np.where(pred>0.5)] = 0
                contours    = cv2.findContours(np.uint8(weights>0)*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

                gaus        = 0 
                for i in range(len(contours)):
                    if cv2.contourArea(contours[i])<20:
                        continue
                    weight  = np.zeros((256,256,3))
                    weight  = cv2.drawContours(weight, contours, i, color=(1,1,1), thickness=-1, lineType=None, hierarchy=None, maxLevel=None, offset=None)
                    weight  = weight[:,:,0]*weights
                    weight  = weight/weight.sum()
                    X, Y    = np.meshgrid(np.arange(256), np.arange(256))
                    ux, uy  = (weight*X).sum(),  (weight*Y).sum()
                    sx, sy  = (weight*(X-ux)**2).sum(), (weight*(Y-uy)**2).sum()
                    sxy     = (weight*(X-ux)*(Y-uy)).sum()
                    gaus    = np.maximum(gaus, gaussian(X, Y, ux, uy, sx/10, sy/10, sxy/10))

                ## fuse = fore + back + gaussian
                fore        = pred*255
                back        = np.zeros_like(fore)
                gaus        = gaus*255
                fuse        = np.stack((fore, gaus, back), axis=-1)
                fuse        = cv2.resize(fuse, (W, H), interpolation=cv2.INTER_LINEAR)

                ## save
                path = self.args.datapath+'/seg/'+name.split('/')[0]
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(self.args.datapath+'/seg/'+name.replace('.jpg', '.png'), np.uint8(fuse))
                if iter%100==0:
                    print(iter)
            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath'    ,default='../dataset/CUB_200_2011')
    parser.add_argument('--snapshot'    ,default='./out/model-32')
    parser.add_argument('--mode'        ,default='val')
    parser.add_argument('--list'        ,default='train.txt')
    parser.add_argument('--clsnum'      ,default=200)
    parser.add_argument('--batch_size'  ,default=1)
    parser.add_argument('--num_workers' ,default=8)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    t = Test(Data, Model, args)
    t.pred_save()