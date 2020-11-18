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
from scipy import stats

class train_dataset(Dataset):
    def __init__(self,root_dir,images_per_classes,classes_per_minibatch,nums_addnegative,transform,cluster=2):
        
        self.transform = transform  # 变换
        self.classes_per_minibatch=classes_per_minibatch
        self.images_per_classes=images_per_classes
        self.minibatch=classes_per_minibatch*images_per_classes
        self.nums_addnegative=nums_addnegative
        self.cluster=cluster

        filename_filted='label_filltered_parent.txt'
        lst=os.listdir(root_dir)
        self.num_all_classes=np.sum(list(map(lambda x: x[-4:]!='.txt', lst)))

        self.data=[[] for i in range(self.num_all_classes)]
        self.label=[]
        self.label_parent_lst_stastic=[[] for i in range(self.num_all_classes)]

        file = open(join(root_dir,filename_filted)) 
        while 1:
            line = file.readline()
            if not line:
                break
            line=line.strip('\n')
            data_l=line.split(',')
            data_npy=data_l[0][:-3]+'npy'
            self.label.append(int(data_l[1]))
            self.data[int(data_l[1])].append(join(root_dir,data_npy))
            # 读取每个子类别 对应的所有父类别           
            self.label_parent_lst_stastic[int(data_l[1])].append(int(data_l[2]))
        file.close()

        # 统计众数,得到每个子类别对应的父类别
        for i in range(len(self.label_parent_lst_stastic)):
            if len(self.label_parent_lst_stastic[i])!=0:
                # print(i,self.label_parent_lst_stastic[i])
                self.label_parent_lst_stastic[i] = stats.mode(self.label_parent_lst_stastic[i])[0][0]#子类:父类
                print(i,self.label_parent_lst_stastic[i])
            else:
                self.label_parent_lst_stastic[i]=-1

        # 统计各个父类别含有的子类别
        self.label_parent_dict={}
        self.label_parent_lst=[]
        for i in range(max(np.array(self.label_parent_lst_stastic))+1):
            self.label_parent_dict[i]=[]
        class_id=0
        for img_id in range(len(self.label_parent_lst_stastic)):
            if self.label_parent_lst_stastic[img_id]!=-1:
                self.label_parent_dict[self.label_parent_lst_stastic[img_id]].append(img_id)#父类:子类
                class_id+=1

        # 各个父类别含有子类别 用list存储
        dict_tmp=self.label_parent_dict.copy()
        for key in dict_tmp.keys():
            if len(dict_tmp)==0:
                del self.label_parent_dict[key]
            else:
                self.label_parent_lst.append(self.label_parent_dict[key])#[[各子类1],[各子类2]]
        self.num_group=int(class_id/self.cluster)

        # self.steps_all=int(len(self.data)/classes_per_minibatch)
        self.steps_all=int(class_id/classes_per_minibatch)
        self.read_order=[]
        self.shuffle()
        self.all_class_ids=list(range(self.num_all_classes))
        
        
    def shuffle(self):#每运行完一个epoch,运行该函数打乱顺序
        read_order_parent=random.sample(range(0,len(self.label_parent_lst)),len(self.label_parent_lst))
        classes_all_tmp=[self.label_parent_lst[i] for i in read_order_parent]
        classes_all=[]
        for tmp in classes_all_tmp:
            classes_all=classes_all+tmp
        read_order_tmp=[]
        for i in range(self.num_group-1):
            read_order_tmp.append(classes_all[2*i:2*i+2])
        read_order_tmp.append(classes_all[2*self.num_group-2:])
        read_order_group=random.sample(range(0,len(read_order_tmp)),len(read_order_tmp))
        read_order_tmp=[read_order_tmp[i] for i in read_order_group]
        self.read_order=[]
        for tmp in read_order_tmp:
            self.read_order=self.read_order+tmp
         

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
        # print(self.steps_all,len(self.read_order))
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