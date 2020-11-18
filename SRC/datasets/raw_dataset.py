
import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os


import cv2 as cv


#和这里的test_dataset全部来自train_data中，只是随机抽取percentage_test_while_training 的图片作为测试集其余作为训练集合
#用于闭集测试，训练集和测试集中是相同的类别，但是不同的样本




class train_dataset_sequence(Dataset):

    def __init__(self,data_dir,transform,percentage_test_while_training=0.05):
        self.data_dir=data_dir
        self.transform = transform  # 变换


        labels = pd.read_csv(self.data_dir+"/label_filltered.txt", header=None)

        self.total_data= labels
        self.n_class=labels.max()[1]+1



    def __len__(self):
        return len(self.total_data)


    def __getitem__(self, index):

        label = self.total_data[1][index]  # 根据索引index获取该图片
        img_path = os.path.join(self.data_dir, self.total_data[0][index])  # 获取索引为index的图片的路径名


        img = cv.imread(img_path)
        sample = {"image": img, "target": label}
        if self.transform:
            sample = self.transform(sample)  # 对样本进行变换
        img, label = sample['image'], sample['target']

        return img,label


class train_dataset(Dataset):

    def __init__(self,data_dir,transform,percentage_test_while_training=0.05):
        self.data_dir=data_dir
        self.transform = transform  # 变换


        labels = pd.read_csv(self.data_dir+"/label_filltered.txt", header=None)
        labels_shuffled=labels.sample(frac=1,random_state=123).reset_index(drop=True) #固定随机种子,使得每次训练都是固定的图片被作为测试集
        self.total_data= labels_shuffled[int(percentage_test_while_training * len(labels_shuffled)):].reset_index(drop=True)
        self.n_class=labels.max()[1]+1



    def __len__(self):
        return len(self.total_data)


    def __getitem__(self, index):

        label = self.total_data[1][index]  # 根据索引index获取该图片
        img_path = os.path.join(self.data_dir, self.total_data[0][index])  # 获取索引为index的图片的路径名


        img = cv.imread(img_path)
        sample = {"image": img, "target": label}
        if self.transform:
            sample = self.transform(sample)  # 对样本进行变换
        img, label = sample['image'], sample['target']

        return img,label


class test_dataset(train_dataset):
    def __init__(self,data_dir,transform,percentage_test_while_training=0.05):
        self.data_dir=data_dir
        self.transform = transform  # 变换


        labels = pd.read_csv(self.data_dir+"/label_filltered.txt", header=None)
        labels_shuffled=labels.sample(frac=1,random_state=123).reset_index(drop=True)
        #self.train_label=labels_shuffled[int(percentage_test_while_training*len(labels_shuffled)):].reset_index(drop=True)
        self.total_data=labels_shuffled[:int(percentage_test_while_training*len(labels_shuffled))]
        self.n_class=labels.max()[1]+1




if __name__ == '__main__':
    aa = train_dataset("/mnt/home/yufei/HWdata/train_data",[],0)
    len(aa)
    for a,b in aa:
        #print(a.size)
        pass
    bb=aa[0]
    b=1



