# -*- coding: utf-8 -*-


import time
import numpy as np
import random
import os
from os.path import join,dirname,realpath
from torch.utils.data import Dataset
import cv2 as cv
import torch


class train_dataset(Dataset):
    def __init__(self,root_dir,images_per_classes,classes_per_minibatch,transform):
        
        self.transform = transform  # 变换
        self.classes_per_minibatch=classes_per_minibatch
        self.images_per_classes=images_per_classes
        self.minibatch=classes_per_minibatch*images_per_classes

        filename_filted='label_filltered.txt'
        lst=os.listdir(root_dir)
        self.num_all_classes=np.sum(list(map(lambda x: x[-4:]!='.txt', lst)))

        self.data=[[] for i in range(self.num_all_classes)]
        self.label=[]

        file = open(join(root_dir,filename_filted)) 
        while 1:
            line = file.readline()
            if not line:
                break
            line=line.strip('\n')
            data_l=line.split(',')
            data_npy=data_l[0]
            self.label.append(int(data_l[1]))
            self.data[int(data_l[1])].append(join(root_dir,data_npy))
        file.close()

        self.steps_all=int(len(self.data)/classes_per_minibatch)
        # self.read_order=random.sample(range(0,self.num_all_classes),self.num_all_classes)
        # self.shuffle()
        
    def shuffle(self):#每运行完一个epoch,运行该函数打乱顺序
        self.read_order=random.shuffle(self.read_order)
        for class_id in range(len(self.data)):
            self.data[class_id]=random.shuffle(self.data[class_id])   

    def __len__(self):#获取总的mini_batch数目
        return self.steps_all
        
    def get_item(self,class_id,img_id):
        img = cv.imread(self.data[class_id][img_id])
        sample = {"image": img}
        if self.transform:
            sample = self.transform(sample)  # 对样本进行变换
        img = sample['image']
        img=img.unsqueeze(0)
        return img

    def __getitem__(self,step):#获取第step个minibatch
        if step>self.steps_all-1:
            print('step_train out of size')
            return
        # class_ids=self.read_order[step*self.classes_per_minibatch:(step+1)*self.classes_per_minibatch]
        class_ids=random.sample(range(0,self.num_all_classes),self.classes_per_minibatch)
        
        start=True
        labels=[]

        for class_id in class_ids:
            num=min(self.classes_per_minibatch,len(self.data[class_id]))
            if num<2:
                print('第{}类图片数目不够'.format(class_id))
                continue
            img_ids=random.sample(range(0,len(self.data[class_id])),num)
            for img_id in img_ids:
                
                img_tmp=self.get_item(class_id,img_id)
                labels.append(class_id)
                # with open('/home/yufei/HUW/models/trip_loss/log/tmp_data.txt',"a") as file:   #”w"代表着每次运行都覆盖内容
                #     file.write(self.data[class_id][img_id]
                #                 +'  label:'+str(class_id)+'\n')
                if start:
                    imgs=img_tmp.detach().clone()
                    start=False
                else:
                    imgs=torch.cat((imgs,img_tmp),dim=0)

        labels=torch.tensor(labels)
        labels=labels.int()
        imgs=imgs

        # with open('/home/yufei/HUW/models/trip_loss/log/tmp_data.txt',"a") as file:   #”w"代表着每次运行都覆盖内容
        #     file.write('===========================================\n')
        return imgs,labels
        
    
