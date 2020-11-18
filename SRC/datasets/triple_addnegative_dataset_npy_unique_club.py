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
from sklearn.preprocessing import LabelEncoder

RANDOM_SEED=222
class train_dataset(Dataset):
    def __init__(self,root_dir,images_per_classes,classes_per_minibatch,nums_addnegative,transform):
        
        self.transform = transform  # 变换
        self.classes_per_minibatch=classes_per_minibatch
        self.images_per_classes=images_per_classes
        self.minibatch=classes_per_minibatch*images_per_classes
        self.nums_addnegative=nums_addnegative


        filename_filted='label_filltered_club.txt'
        labels = pd.read_csv(root_dir+"/"+filename_filted, header=None)




        self.dataset_data=labels

        to_encode=list(self.dataset_data[1])

        le=LabelEncoder()
        le.fit(to_encode)
        to_encode=le.transform(to_encode)
        self.dataset_data[1]=to_encode
        self.n_class= len(le.classes_)


        self.num_all_classes=self.dataset_data[1].max()+1
        self.data=[[] for i in range(self.num_all_classes)]
        self.label=[]
        for i in range(len(self.dataset_data[0])):
            data_npy=self.dataset_data[0][i][:-3]+'npy'
            self.label.append(int(self.dataset_data[1][i]))
            self.data[int(self.dataset_data[1][i])].append(join(root_dir,data_npy))


        self.steps_all=int(len(self.data)/classes_per_minibatch)
        self.read_order=random.sample(range(0,self.num_all_classes),self.num_all_classes)
        self.all_class_ids=list(range(self.num_all_classes))
        
    def shuffle(self):#每运行完一个epoch,运行该函数打乱顺序
        self.read_order=random.sample(range(0,self.num_all_classes),self.num_all_classes)
        # for class_id in range(len(self.data)):
        #     self.data[class_id]=random.shuffle(self.data[class_id])   

    def __len__(self):#获取总的mini_batch数目
        return self.steps_all
        
    def get_item(self,class_id,img_id):
        img = np.load(join(self.data[class_id][img_id]))
        sample = {"image": img}
        if self.transform:
            sample = self.transform(sample)  # 对样本进行变换
        img = sample['image']

        # debug
        # img=torch.tensor(img)

        img=img.unsqueeze(0)
        return img

    def __getitem__(self,step):#获取第step个minibatch
        if step>self.steps_all-1:
            print('step_train out of size')
            return

        class_ids=self.read_order[step*self.classes_per_minibatch:(step+1)*self.classes_per_minibatch]
        # class_ids=random.sample(range(0,self.num_all_classes),self.classes_per_minibatch)
        
        
        start=True
        labels=[]

        for class_id in class_ids:
            num=min(self.images_per_classes,len(self.data[class_id]))
            # if num<2:
            #     print('第{}类图片数目不够'.format(class_id))
            #     continue
            img_ids=random.sample(range(0,len(self.data[class_id])),num)
            for img_id in img_ids:
                
                img_tmp=self.get_item(class_id,img_id)
                labels.append(class_id)
                if start:
                    imgs=img_tmp
                    start=False
                else:
                    imgs=torch.cat((imgs,img_tmp),dim=0)
                    
        if self.nums_addnegative !=0:
            negative_class_ids=[[i for i in self.all_class_ids if i not in class_ids]]
            negative_class_ids=random.sample(negative_class_ids,self.nums_addnegative)
            for class_id in negative_class_ids:
                num=min(self.images_per_classes,len(self.data[class_id]))
                # if num<2:
                    # print('第{}类图片数目不够'.format(class_id))
                    # continue
                img_ids=random.sample(range(0,len(self.data[class_id])),2)
                for img_id in img_ids:
                    
                    img_tmp=self.get_item(class_id,img_id)
                    labels.append(class_id)
                    imgs=torch.cat((imgs,img_tmp),dim=0)


        labels=torch.tensor(labels)
        labels=labels.int()
        imgs=imgs

        return imgs,labels
        
    

if __name__ == '__main__':
    
    tr=train_dataset("/home/yufei/HUW3/data/train_data_resize512_rgb",\
                images_per_classes=10, \
                     nums_addnegative=0,\
                classes_per_minibatch=10,\
                transform=None)


    tr.steps_all=10
    n=0


        
