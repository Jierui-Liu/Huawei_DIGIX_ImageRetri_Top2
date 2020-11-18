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
    def __init__(self,root_dir,images_per_classes,classes_per_minibatch,transform,percentage_test_while_training=0.10):
        
        # self.transform = transform  # 变换
        self.classes_per_minibatch=classes_per_minibatch
        self.images_per_classes=images_per_classes
        self.minibatch=classes_per_minibatch*images_per_classes

        # filename_filted='label_filltered.txt'
        # lst=os.listdir(root_dir)
        # self.num_all_classes=np.sum(list(map(lambda x: x[-4:]!='.txt', lst)))

        # self.data=[[] for i in range(self.num_all_classes)]
        # self.label=[]


         #percentage_test_while_training使用多少比例的数据作为测试集
        self.data_dir=root_dir
        self.transform = transform  # 变换

        labels = pd.read_csv(self.data_dir+"/label_filltered.txt", header=None)
        self.total_number_of_class=labels.max()[1]+1
        labels_shuffled=pd.DataFrame(list(range(self.total_number_of_class))).sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)

        num_of_train_class=int(self.total_number_of_class*(1-percentage_test_while_training))  #2787
        list_dataset_for_train=[]

        for i in range(num_of_train_class):
            list_dataset_for_train.append(labels[labels[1]==labels_shuffled[0][i]].reset_index(drop=True))

        self.dataset_data=pd.concat(list_dataset_for_train).reset_index(drop=True) #2487elf.dataset_data=pd.concat(list_dataset_for_train).reset_index(drop=True) #2487
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
        print(len( self.data),self.num_all_classes,self.classes_per_minibatch)
        
    def shuffle(self):#每运行完一个epoch,运行该函数打乱顺序
        self.read_order=random.shuffle(self.read_order)
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

        # class_ids=self.read_order[step*self.classes_per_minibatch:(step+1)*self.classes_per_minibatch]
        class_ids=random.sample(range(0,self.num_all_classes),self.classes_per_minibatch)
        
        
        start=True
        labels=[]
        imgs=[]

        for class_id in class_ids:
            num=min(self.images_per_classes,len(self.data[class_id]))
            # print('num',num)
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

        labels=torch.tensor(labels[1:])
        labels=labels.int()
        # print(imgs.shape )
        imgs=imgs[1:,...]

        # with open('/home/yufei/HUW/models/trip_loss/log/tmp_data.txt',"a") as file:   #”w"代表着每次运行都覆盖内容
        #     file.write('===========================================\n')
        return imgs,labels
        
    

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