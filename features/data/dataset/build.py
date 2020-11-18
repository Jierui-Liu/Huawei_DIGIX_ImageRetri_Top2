'''
@Author      : now more
@Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
@Description : 
LastEditTime: 2020-09-24 11:02:57
'''

from torch.utils.data import Dataset
import cv2 as cv
import pandas as pd
import os
import os.path as osp
import numpy as np
import torch
from os.path import  join
import random
from collections import defaultdict
from copy import deepcopy

class load_testB(Dataset):
    def __init__(self,image_dir,label_file,transforms=None):
        self.image_dir = image_dir
        self.label = pd.read_csv(label_file,header=None).to_numpy()
        self.transforms = transforms
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,idx):
        image_path,target = self.label[idx,:].tolist()
        image = cv.imread(osp.join(self.image_dir,image_path))
        image_name = image_path
        image_name = image_name.replace(".npy",".jpg")

        sample = {"image":image,"target":target}
        if self.transforms:
            sample = self.transforms(sample)
        image,target = sample['image'],sample['target']
        # print(image_name)
        return image,image_name

        
class load_image_per_instance(Dataset):
    def __init__(self,image_dir,images_per_instance=4,transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.images_per_instance = images_per_instance
        self.label = pd.read_csv(osp.join(image_dir,"label.txt"),header=None).to_numpy()
        
        self.instance_dict = defaultdict(list)
        for i in range(len(self.label)):
            image_path,image_id = self.label[i]
            self.instance_dict[image_id].append(i)
        self.instance_ids = self.instance_dict.keys()
    
    def __len__(self):
        return len(self.instance_ids)
    
    def __getitem__(self,idx):
        idxs = deepcopy(self.instance_dict[idx])
        num = len(idxs)
        while(num<2):
            idx = np.random.randint(0,len(self.instance_ids)) #重选id
            idxs = deepcopy(self.instance_dict[idx])
            num = len(idxs)

        idxs = np.random.choice(idxs,size=self.images_per_instance,replace=True)
        np.random.shuffle(idxs)

        batch_path = [osp.join(self.image_dir,image_path) for image_path in self.label[idxs,0]]
        print(batch_path)

        batch_images = [cv.imread(osp.join(self.image_dir,image_path)) for image_path in self.label[idxs,0]]
        # convert BGR2RGB
        batch_images = [cv.cvtColor(image,cv.COLOR_BGR2RGB) for image in batch_images]
        batch_targets = self.label[idxs,1].tolist()
        batch_samples = [{"image":image,"target":target} for (image,target) in zip(batch_images,batch_targets)]
        if self.transforms:
            batch_samples = [self.transforms(sample) for sample in batch_samples]
        
        batch_images,batch_targets = list(),list()
        for sample in batch_samples:
            batch_images.append(sample['image'].unsqueeze(0))
            batch_targets.append(sample['target'])
        
        batch_images = torch.cat(batch_images,dim=0)
        batch_targets = torch.tensor(batch_targets,dtype=batch_images.dtype)
        return batch_images,batch_targets

class load_npy_per_instance(Dataset):
    def __init__(self,image_dir,images_per_instance=4,transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.images_per_instance = images_per_instance
        self.label = pd.read_csv(osp.join(image_dir,"label.txt"),header=None).to_numpy()
        
        self.instance_dict = defaultdict(list)
        for i in range(len(self.label)):
            image_path,image_id = self.label[i]
            self.instance_dict[image_id].append(i)
        self.instance_ids = self.instance_dict.keys()
    
    def __len__(self):
        return len(self.instance_ids)
    
    def __getitem__(self,idx):
        idxs = deepcopy(self.instance_dict[idx])
        num = len(idxs)
        while(num<1):
            idx = np.random.randint(0,len(self.instance_ids)) #重选id
            idxs = deepcopy(self.instance_dict[idx])
            num = len(idxs)

        idxs = np.random.choice(idxs,size=self.images_per_instance,replace=True)
        np.random.shuffle(idxs)

        batch_images = [np.load(osp.join(self.image_dir,image_path.replace(".jpg",".npy"))) for image_path in self.label[idxs,0]]
        batch_targets = self.label[idxs,1].tolist()
        batch_samples = [{"image":image,"target":target} for (image,target) in zip(batch_images,batch_targets)]
        if self.transforms:
            batch_samples = [self.transforms(sample) for sample in batch_samples]
        
        batch_images,batch_targets = list(),list()
        for sample in batch_samples:
            batch_images.append(sample['image'].unsqueeze(0))
            batch_targets.append(sample['target'])
        
        batch_images = torch.cat(batch_images,dim=0)
        batch_targets = torch.tensor(batch_targets,dtype=batch_images.dtype)
        return batch_images,batch_targets
        
class load_dataAll(Dataset):
    def __init__(self,image_dir,label_file,transforms=None):
        self.image_dir = image_dir
        self.label = pd.read_csv(label_file,header=None).to_numpy()
        self.transforms = transforms
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,idx):
        image_path,target = self.label[idx,:].tolist()
        image = cv.imread(osp.join(self.image_dir,image_path))
        
        sample = {"image":image,"target":target}
        if self.transforms:
            sample = self.transforms(sample)
        image,target = sample['image'],sample['target']
        return image,target
        
class load_npyAll(Dataset):
    def __init__(self,image_dir,label_file,transforms=None):
        self.image_dir = image_dir
        self.label = pd.read_csv(label_file,header=None).to_numpy()
        self.transforms = transforms
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,idx):
        image_path,target = self.label[idx,:].tolist()
        image_path = image_path.replace(".jpg",".npy")

        image = np.load(osp.join(self.image_dir,image_path))
        
        sample = {"image":image,"target":target}
        if self.transforms:
            sample = self.transforms(sample)
        image,target = sample['image'],sample['target']

        return image,target


class load_image(Dataset):
    def __init__(self,image_dir,transforms):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_list = os.listdir(image_dir)
    
        self.image_list.sort()
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,idx):
        image_name = self.image_list[idx]

        image = cv.imread(osp.join(self.image_dir,image_name))
        sample = {'image':image}
        if self.transforms:
            sample = self.transforms(sample)
        image = sample['image']
        return image,image_name

class load_npy(Dataset):
    def __init__(self,image_dir,transforms):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_list = os.listdir(image_dir)
    
        self.image_list.sort()
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,idx):
        image_name = self.image_list[idx]
        image_name = image_name.replace(".jpg",".npy")

        image = np.load(osp.join(self.image_dir,image_name))
        sample = {'image':image}
        if self.transforms:
            sample = self.transforms(sample)
        image = sample['image']
        image_name = image_name.replace(".npy",".jpg")
        return image,image_name

class train_dataset(Dataset):
    def __init__(self,root_dir,images_per_classes,classes_per_minibatch,transforms):
        
        self.transform = transforms  # 变换
        self.classes_per_minibatch=classes_per_minibatch
        self.images_per_classes=images_per_classes
        self.minibatch=classes_per_minibatch*images_per_classes

        filename_filted='label.txt'
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
            data_npy=data_l[0][:-3]+'npy'
            self.label.append(int(data_l[1]))
            self.data[int(data_l[1])].append(join(root_dir,data_npy))
        file.close()

        self.steps_all=int(len(self.data)/classes_per_minibatch)
        self.read_order=random.sample(range(0,self.num_all_classes),self.num_all_classes)
        
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

        # print(len(self.read_order))
        class_ids=self.read_order[step*self.classes_per_minibatch:(step+1)*self.classes_per_minibatch]
        # class_ids=random.sample(range(0,self.num_all_classes),self.classes_per_minibatch)
        
        
        start=True
        labels=[]

        for class_id in class_ids:
            num=min(self.images_per_classes,len(self.data[class_id]))
            # if num<2:
            #     # print('第{}类图片数目不够'.format(class_id))
            #     continue
            while(num<2):
                class_id = np.random.choice(3097,1)[0]
                num = len(self.data[class_id])
            img_ids = np.random.choice(len(self.data[class_id]),self.images_per_classes)
            for img_id in img_ids:
                
                img_tmp=self.get_item(class_id,img_id)
                labels.append(class_id)
                if start:
                    imgs=img_tmp.detach().clone()
                    start=False
                else:
                    imgs=torch.cat((imgs,img_tmp),dim=0)

        labels=torch.tensor(labels)
        labels=labels.int()


        # with open('/home/yufei/HUW/models/trip_loss/log/tmp_data.txt',"a") as file:   #”w"代表着每次运行都覆盖内容
        #     file.write('===========================================\n')
        return imgs,labels


if __name__ == "__main__":
    # dataset = load_dataAll(image_dir=r"/Users/linhonghui/Desktop/train_data",label_file=r"/Users/linhonghui/Desktop/train_data/label.txt")
    dataset = load_npyAll(image_dir = r"/home/LinHonghui/Datasets/HW_ImageRetrieval/train_data_resize384",
                    label_file = r"/home/LinHonghui/Datasets/HW_ImageRetrieval/train_data_resize384/label.txt", )
    print(len(dataset))
    import pdb; pdb.set_trace()