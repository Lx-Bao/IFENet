#coding=utf-8

import os
import cv2
import numpy as np
import torch
try:
    from . import transform
except:
    import transform
from torch.utils.data import Dataset
#BGR
mean_rgb = np.array([[[0.391*255, 0.363*255, 0.338*255]]])
mean_t =np.array([[[0.170*255,  0.403*255, 0.556*255]]])
mean_d =np.array([[[0.034*255,  0.034*255, 0.034*255]]])
std_rgb = np.array([[[0.224 * 255, 0.217 * 255, 0.206 * 255]]])
std_t = np.array([[[0.160 * 255, 0.188 * 255, 0.238 * 255]]])
std_d = np.array([[[0.007 * 255, 0.007 * 255, 0.007 * 255]]])

def getRandomSample(rgb,t,d):
    n = np.random.randint(10)
    zero = np.random.randint(2)
    if n==1:
        if zero:
            rgb = torch.from_numpy(np.zeros_like(rgb))
        else:
            rgb = torch.from_numpy(np.random.randn(*rgb.shape))
    elif n==2:
        if zero:
            t = torch.from_numpy(np.zeros_like(t))
            d = torch.from_numpy(np.zeros_like(d))

        else:
            t = torch.from_numpy(np.random.randn(*t.shape))
            d = torch.from_numpy(np.random.randn(*d.shape))
    return rgb,t,d

class Data(Dataset):
    def __init__(self, root, mode='train'):
        self.samples = []
        lines = os.listdir(os.path.join(root, 'GT'))
        self.mode = mode
        for line in lines:
            rgbpath = os.path.join(root, 'V', line[:-4]+'.png')
            tpath = os.path.join(root, 'T', line[:-4]+'.png')
            dpath = os.path.join(root, 'D', line[:-4] + '.png')
            egpath = os.path.join(root, 'eg', line[:-4] + '.png')
            maskpath = os.path.join(root, 'GT', line)
            self.samples.append([rgbpath,tpath,dpath,egpath,maskpath])

        if mode == 'train':
            self.transform = transform.Compose(transform.Normalize(mean1=mean_rgb, mean2=mean_t, mean3=mean_d, std1=std_rgb, std2=std_t, std3=std_d),
                                                transform.Resize(384, 384), transform.Random_rotate(),
                                                transform.RandomHorizontalFlip(), transform.ToTensor())

        elif mode == 'test':
            self.transform = transform.Compose(transform.Normalize(mean1=mean_rgb, mean2=mean_t, mean3=mean_d, std1=std_rgb, std2=std_t, std3=std_d),
                                                transform.Resize(384, 384),
                                                transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        rgbpath,tpath,dpath,egpath,maskpath = self.samples[idx]
        rgb = cv2.imread(rgbpath).astype(np.float32)
        t = cv2.imread(tpath).astype(np.float32)
        d = cv2.imread(dpath).astype(np.float32)
        eg = cv2.imread(egpath).astype(np.float32)
        mask = cv2.imread(maskpath).astype(np.float32)
        H, W, C = mask.shape
        rgb,t,d,eg,mask = self.transform(rgb,t,d,eg,mask)
        if  self.mode == 'train':
            rgb,t,d = getRandomSample(rgb,t,d)
        return rgb,t,d,eg,mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)
