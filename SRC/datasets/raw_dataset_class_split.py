
import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os
import numpy as np
import cv2 as cv
import random
from sklearn.preprocessing import LabelEncoder

#训练集中一共3094类，编号0-3096
#其中缺失：
#1453
#2284
#1639


RANDOM_SEED=222
class train_dataset(Dataset):

    def __init__(self,data_dir,transform,percentage_test_while_training=0.10):
        #percentage_test_while_training使用多少比例的数据作为测试集
        self.data_dir=data_dir
        self.transform = transform  # 变换


        labels = pd.read_csv(self.data_dir+"/label_filltered.txt", header=None)
        self.total_number_of_class=labels.max()[1]+1
        labels_shuffled=pd.DataFrame(list(range(self.total_number_of_class))).sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)

        num_of_train_class=int(self.total_number_of_class*(1-percentage_test_while_training))  #2787
        list_dataset_for_train=[]

        for i in range(num_of_train_class):
            list_dataset_for_train.append(labels[labels[1]==labels_shuffled[0][i]].reset_index(drop=True))

        self.dataset_data=pd.concat(list_dataset_for_train).reset_index(drop=True) #2487
        to_encode=list(self.dataset_data[1])

        le=LabelEncoder()
        le.fit(to_encode)
        to_encode=le.transform(to_encode)
        self.dataset_data[1]=to_encode
        self.n_class= len(le.classes_)



    def __len__(self):
        return len(self.dataset_data)


    def __getitem__(self, index):

        label = self.dataset_data[1][index]  # 根据索引index获取该图片
        img_path = os.path.join(self.data_dir, self.dataset_data[0][index][:-3] + "npy")  # 获取索引为index的图片的路径名


        img = np.load(img_path)
        sample = {"image": img, "target": label}
        if self.transform:
            sample = self.transform(sample)  # 对样本进行变换
        return sample['image'], sample['target']


class query_dataset(train_dataset):
    def __init__(self,data_dir,transform,percentage_test_while_training=0.10,percentage_for_query=0.2):
        #percentage_test_while_training 与train_dataset 保持一致
        #percentage_for_query加上gallery_dataset中的percentage_for_query 一致
        self.data_dir=data_dir
        self.transform = transform  # 变换
        labels = pd.read_csv(self.data_dir+"/label_filltered.txt", header=None)
        self.total_number_of_class=labels.max()[1]+1
        num_of_train_class=int(self.total_number_of_class*(1-percentage_test_while_training))

        labels_shuffled=pd.DataFrame(list(range(self.total_number_of_class))).sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)

        list_dataset_for_query=[]

        for i in range(num_of_train_class,len(labels_shuffled)):
            label=labels_shuffled[0][i]
            sample_in_class=labels[labels[1]==label].reset_index(drop=True).sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)
            number_of_sample_query=max(int(percentage_for_query*len(sample_in_class)),1)
            list_dataset_for_query.append(sample_in_class[:number_of_sample_query])
        self.dataset_data=pd.concat(list_dataset_for_query).reset_index(drop=True)



class gallery_dataset(train_dataset):
    def __init__(self, data_dir, transform, percentage_test_while_training=0.10, percentage_for_query=0.2):
        self.data_dir=data_dir
        self.transform = transform  # 变换
        labels = pd.read_csv(self.data_dir+"/label_filltered.txt", header=None)
        self.total_number_of_class=labels.max()[1]+1
        num_of_train_class=int(self.total_number_of_class*(1-percentage_test_while_training))

        labels_shuffled=pd.DataFrame(list(range(self.total_number_of_class))).sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)

        list_dataset_for_gallery=[]

        for i in range(num_of_train_class,len(labels_shuffled)):
            label=labels_shuffled[0][i]
            sample_in_class=labels[labels[1]==label].reset_index(drop=True).sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)
            number_of_sample_query=max(int(percentage_for_query*len(sample_in_class)),1)
            list_dataset_for_gallery.append(sample_in_class[number_of_sample_query:])
        self.dataset_data=pd.concat(list_dataset_for_gallery).reset_index(drop=True)






if __name__ == '__main__':
    #td = train_dataset("/mnt/home/yufei/HWdata/train_data_resize384",[])
    qd=query_dataset("/mnt/home/yufei/HWdata/train_data_resize384",[])
    #gd=gallery_dataset("/mnt/home/yufei/HWdata/train_data_resize384",[])



        #train #query #gallery



    for a,b in qd:
        pass


    b=1

# train_class = random.sample(range(0, self.total_number_of_class), self.numer_of_class_for_train)
# labels[1].isin([0, 1, 2, 3, 4])