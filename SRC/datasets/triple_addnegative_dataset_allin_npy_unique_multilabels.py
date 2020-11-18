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

def img_stitch(imgs_to_stitch):
    n=len(imgs_to_stitch)
    h,w,c=imgs_to_stitch[0].shape
    w_center=int(w/2)
    h_center=int(h/2)
    if n==2:
        if np.random.random() <= 0.5:
            half_w_crop=int(w*3/10)
            overlap=half_w_crop*4-w
            overlap_start=-half_w_crop*2+w
            imgs_to_stitch_crop=[img[:,w_center-half_w_crop:w_center+half_w_crop,:] for img in imgs_to_stitch]

            img=np.zeros((h,w,3))
            img[:,:half_w_crop*2]=imgs_to_stitch_crop[0]
            img[:,-half_w_crop*2:]=imgs_to_stitch_crop[1]
            img[:,overlap_start:overlap_start+overlap]/=2
            img=img.astype(np.uint8)
        else:
            half_h_crop=int(h*3/10)
            overlap=half_h_crop*4-w
            overlap_start=-half_h_crop*2+w
            imgs_to_stitch_crop=[img[h_center-half_h_crop:h_center+half_h_crop,:,:] for img in imgs_to_stitch]

            img=np.zeros((h,w,3))
            img[:half_h_crop*2,:]=imgs_to_stitch_crop[0]
            img[-half_h_crop*2:,:]=imgs_to_stitch_crop[1]
            img[overlap_start:overlap_start+overlap,:]/=2
            img=img.astype(np.uint8)

    return img

class train_dataset(Dataset):
    def __init__(self,root_dir,images_per_classes,classes_per_minibatch,nums_addnegative,transform,transform_stitch,num_labels=2):
        
        self.transform = transform  # 变换
        self.transform_stitch = transform_stitch  # 变换
        self.classes_per_minibatch=classes_per_minibatch-1
        self.images_per_classes=images_per_classes
        self.minibatch=self.classes_per_minibatch*images_per_classes
        self.nums_addnegative=nums_addnegative
        self.num_labels=num_labels

        filename_filted='label_filltered.txt'
        lst=os.listdir(root_dir)
        self.num_all_classes=np.sum(list(map(lambda x: x[-4:]!='.txt', lst)))

        self.data=[[] for i in range(self.num_all_classes)]
        self.data_name=[[] for i in range(self.num_all_classes)]
        self.label=[]

        file = open(join(root_dir,filename_filted)) 
        while 1:
            line = file.readline()
            if not line:
                break
            line=line.strip('\n')
            data_l=line.split(',')
            data_npy=data_l[0][:-3]+'npy'
            self.label.append(int(data_l[1]))
            self.data_name[int(data_l[1])].append(join(root_dir,data_npy))
        file.close()


        for class_id in range(len(self.data_name)):
            start=True
            for img_id in range(len(self.data_name[class_id])):
                img_tmp=np.load(self.data_name[class_id][img_id])#[np.newaxis,...]
                if start:
                    # self.data[class_id]=img_tmp
                    self.data[class_id]=np.zeros((len(self.data_name[class_id]),img_tmp.shape[0],img_tmp.shape[1],img_tmp.shape[2]\
                                            )).astype(np.uint8)
                    self.data[class_id][img_id,...]=img_tmp
                    start=False
                else:
                    self.data[class_id][img_id,...]=img_tmp


        self.steps_all=int(len(self.data)/classes_per_minibatch)
        self.read_order=random.sample(range(0,self.num_all_classes),self.num_all_classes)
        self.all_class_ids=list(range(self.num_all_classes))
        
    def shuffle(self):#每运行完一个epoch,运行该函数打乱顺序
        self.read_order=random.sample(range(0,self.num_all_classes),self.num_all_classes)

        
    def __len__(self):#获取总的mini_batch数目
        return self.steps_all

    def __getitem__(self,step):#获取第step个minibatch
        if step>self.steps_all-1:
            print('step_train out of size')
            return

        class_ids=self.read_order[step*self.classes_per_minibatch:(step+1)*self.classes_per_minibatch]

        
        
        start=True
        labels=[]
        tmp_imgs=[]
        tmp_labels=[]

        for class_id in class_ids:
            num=min(self.images_per_classes,len(self.data[class_id]))
            img_ids=random.sample(range(0,len(self.data[class_id])),num)
            if num>0:
                tmp_imgs.append(self.data[class_id][img_ids[0]])
                tmp_labels.append(class_id)
            for img_id in img_ids:
                
                img_tmp=self.data[class_id][img_id]
                sample = {"image": img_tmp}
                if self.transform:
                    sample = self.transform(sample)  # 对样本进行变换
                img_tmp = sample['image'].unsqueeze(0)

                label=np.array([1]+[class_id for i in range(self.num_labels)])
                labels.append(label)
                if start:
                    imgs=img_tmp.detach()
                    start=False
                else:
                    imgs=torch.cat((imgs.detach(),img_tmp),dim=0)

        if self.nums_addnegative !=0:
            negative_class_ids=[[i for i in self.all_class_ids if i not in class_ids]]
            negative_class_ids=random.sample(negative_class_ids,self.nums_addnegative)
            for class_id in negative_class_ids:
                num=min(self.images_per_classes,len(self.data[class_id]))
                img_ids=random.sample(range(0,len(self.data[class_id])),2)
                tmp_imgs.append(self.data[class_id][img_ids[0]])
                tmp_labels.append(class_id)
                for img_id in img_ids:
                    img_tmp=self.data[class_id][img_id]
                    sample = {"image": img_tmp}
                    if self.transform:
                        sample = self.transform(sample)  # 对样本进行变换
                    img_tmp = sample['image'].unsqueeze(0)

                    label=np.array([1]+[class_id for i in range(self.num_labels)])
                    labels.append(label)
                    imgs=torch.cat((imgs,img_tmp),dim=0)

        for i in range(self.images_per_classes):
            img_ids=random.sample(range(0,len(tmp_imgs)),self.num_labels)
            imgs_to_stitch=[tmp_imgs[img_id] for img_id in img_ids]
            labels_to_stitch=[tmp_labels[img_id] for img_id in img_ids]

            # print(imgs_to_stitch[0].shape)
            # print(imgs_to_stitch[1].shape)
            # print('=====================')
                    
            label=np.array([self.num_labels]+[class_id for class_id in labels_to_stitch])
            labels.append(label)
            img_tmp=img_stitch(imgs_to_stitch)
            sample = {"image": img_tmp}
            if self.transform_stitch:
                sample = self.transform_stitch(sample)  # 对样本进行变换
            img_tmp = sample['image'].unsqueeze(0)
            imgs=torch.cat((imgs,img_tmp),dim=0)


                    
        labels=torch.tensor(labels)
        labels=labels.int()
        imgs=imgs

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