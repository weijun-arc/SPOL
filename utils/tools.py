#coding=utf-8

import numpy as np
import torch.nn as nn

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


def IoU(A, B):
    xmin_a, ymin_a, xmax_a, ymax_a = A
    xmin_b, ymin_b, xmax_b, ymax_b = B
    weight = min(xmax_a, xmax_b) - max(xmin_a, xmin_b)
    height = min(ymax_a, ymax_b) - max(ymin_a, ymin_b)
    if weight<=0 or height<=0:
        return 0
    sa    = (xmax_a-xmin_a)*(ymax_a-ymin_a)
    sb    = (xmax_b-xmin_b)*(ymax_b-ymin_b)
    inter = weight*height
    union = sa+sb-inter
    return inter/(union+1e-12)


def gaussian(x, y, ux, uy, sx, sy, sxy, pred=None):
    c   = -1/(2*(1-sxy**2/sx/sy))
    dx  = (x-ux)**2/sx
    dy  = (y-uy)**2/sy
    dxy = (x-ux)*(y-uy)*sxy/sx/sy
    return np.exp(c*(dx-2*dxy+dy))
