# -*- coding: utf-8 -*-


import time
import numpy as np
import random
import os
from os.path import join,dirname,realpath
from torch.utils.data import Dataset
import cv2 as cv
import torch
from torch.utils.data import DataLoader
import torchvision
import pandas as pd
import sys
sys.path.append('/home/yufei/HUW4')
from SRC.pre_process.opencv_functional import *

class train_dataset(Dataset):
    def __init__(self,image_dir,transforms,p=0.4):
        self.image_dir = image_dir
        self.transforms = transforms
        self.p=p
        if os.path.exists(os.path.join(self.image_dir,"label_filltered.txt")):  #判断是否存在噪声类型文件，不存在则默认该文件夹下所有图像不带噪声
            labels = pd.read_csv(os.path.join(self.image_dir,"label_filltered.txt"), header=None)
            self.image_list=list(labels[0])
            self.noise_type=list(labels[1])






    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,idx):
        image_name = self.image_list[idx].replace(".npy",".jpg")
        image = cv.imread(os.path.join(self.image_dir,image_name))[...,::-1].copy()
        # print('======================',image.shape)
        label=torch.Tensor([0])
        sample = {'image':image}
        
        random_s=np.random.random()
        if random_s <= self.p/3:
            label=torch.Tensor([1])
            sample=peppernoise(sample,0.09+(np.random.random()-0.5)/15)
        elif random_s <= self.p and random_s > self.p/3:
            label=torch.Tensor([2])
            sample=gaussiannoise_correction(sample,23+(np.random.random()-0.5)*10)

        if self.transforms:
            sample = self.transforms(sample)
        image = sample['image']
        return image,label
     




    

if __name__ == '__main__':
    
    tr=train_dataset("/mnt/home/yufei/HWdata/train_data_resize320",\
                images_per_classes=10,\
                classes_per_minibatch=10,\
                transform=None)
    train_loader = DataLoader(dataset=tr, batch_size=1, shuffle=True ,num_workers=8)
    print(tr.__len__())
    tr.steps_all=10
    n=0
    for i in train_loader:
        tr.steps_all=5
        n=n+1
        print(n)

        
    tr.steps_all=5
    n=0
    for i in train_loader:
        n=n+1
        print(n)
    print(tr.__len__())
    tr.__getitem__(0)