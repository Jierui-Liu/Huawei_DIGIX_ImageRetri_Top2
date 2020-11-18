# -*- coding: utf-8 -*-


import time
import numpy as np
import random
import os
from os.path import join,dirname,realpath


class triple_dataloader_train_jpg(object):
    def __init__(self,root_dir,images_per_classes,classes_per_minibatch,transform):
        
        self.transform = transform  # 变换
        self.classes_per_minibatch=classes_per_minibatch
        self.images_per_classes=images_per_classes
        self.minibatch=classes_per_minibatch*images_per_classes

        filename_filted='label_filltered.txt'
        lst=os.listdir(root_dir)
        num_all_classes=np.sum(list(map(lambda x: x[-4:]!='.txt', lst))

        self.data=[[] for i in range(num_all_classes)]
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
        self.read_order=random.sample(range(0,num_all_classes),num_all_classes)
        
    def shuffle(self):#每运行完一个epoch,运行该函数打乱顺序
        self.read_order=random.shuffle(self.read_order)
        for class_id in range(len(self.data)):
            self.data[class_id]=random.shuffle(self.data[class_id])   

    def get_steps(self):#获取总的mini_batch数目
        return self.steps_all
        
    def get_item(self,class_id,img_id):
        img = cv.imread(join(self.data[class_id][img_id]))
        img=self.transform(img)
        img=img.unsqueeze(0)
        return img

    def get_batch(self,step):#获取第step个minibatch
        if step>self.steps_all-1:
            print('step_train out of size')
            return
        class_ids=self.read_order[step*self.classes_per_minibatch:(step+1)*self.classes_per_minibatch]
        
        imgs=get_item(class_ids[0],0)
        labels=[0]

        for class_id in class_ids:
            for img_id in range(self.images_per_classes):
                if img_id>=len(self.data[class_id]):
                    img_tmp=get_item(class_id,img_id)
                    imgs=torch.cat((imgs,img_tmp),dim=0)
                    label.append(class_id)
                else:
                    print('第{}类图片数目不够'.format(class_id))
                    break

        labels=torch.tensor(labels[1:])
        labels=labels.int()
        imgs=imgs[1:,...]

        return imgs,labels
        
    

class triple_dataloader_train_npy(object):
    def __init__(self,root_dir,images_per_classes,classes_per_minibatch,transform):
        
        self.transform = transform  # 变换
        self.root_dir=root_dir
        self.classes_per_minibatch=classes_per_minibatch
        self.images_per_classes=images_per_classes
        self.minibatch=classes_per_minibatch*images_per_classes

        filename_filted='label_filltered.txt'
        lst=os.listdir(root_dir)
        num_all_classes=np.sum(list(map(lambda x: x[-4:]!='.txt', lst))

        self.data=[[] for i in range(num_all_classes)]
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
        self.read_order=random.sample(range(0,num_all_classes),num_all_classes)
        
    def shuffle(self):#每运行完一个epoch,运行该函数打乱顺序
        self.read_order=random.shuffle(self.read_order)
        for class_id in range(len(self.data)):
            self.data[class_id]=random.shuffle(self.data[class_id])   

    def get_steps(self):#获取总的mini_batch数目
        return self.steps_all
        
    def get_item(self,class_id,img_id):
        img = np.load(join(self.data[class_id][img_id]))
        img=self.transform(img)
        img=img.unsqueeze(0)
        return img

    def get_batch(self,step):#获取第step个minibatch
        if step>self.steps_all-1:
            print('step_train out of size')
            return
        class_ids=self.read_order[step*self.classes_per_minibatch:(step+1)*self.classes_per_minibatch]
        
        imgs=get_item(class_ids[0],0)
        labels=[0]

        for class_id in class_ids:
            for img_id in range(self.images_per_classes):
                if img_id>=len(self.data[class_id]):
                    img_tmp=get_item(class_id,img_id)
                    imgs=torch.cat((imgs,img_tmp),dim=0)
                    label.append(class_id)
                else:
                    print('第{}类图片数目不够'.format(class_id))
                    break

        labels=torch.tensor(labels[1:])
        labels=labels.int()
        imgs=imgs[1:,...]

        return imgs,labels
        