

import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os


import cv2 as cv



class eval_dataset(Dataset):
    def __init__(self,image_dir,transforms,noise_type_file_path="default"):
        self.image_dir = image_dir
        self.transforms = transforms
        if noise_type_file_path=="default":
            noise_type_file_path=os.path.join(self.image_dir, "noise_type.txt")   #没有指定就默认在data文件夹下面



        if os.path.exists(noise_type_file_path):  #判断是否存在噪声类型文件，不存在则默认该文件夹下所有图像不带噪声
            labels = pd.read_csv(noise_type_file_path, header=None)
            self.image_list=list(labels[0])
            self.noise_type=list(labels[1])

        else:
            print("warning:noise_type.txt not found, assuming no noise")
            self.image_list = os.listdir(image_dir)
            self.image_list.sort()
            self.noise_type=[0]*len(self.image_list)  #没有噪声类型文件，假设所有图片都不含噪声




    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,idx):
        image_name = self.image_list[idx].replace(".npy",".jpg")
        image = cv.imread(os.path.join(self.image_dir,image_name))[...,::-1].copy()
        sample = {'image':image,"label":self.noise_type[idx]}

        if self.transforms:
            sample = self.transforms(sample)
        image = sample['image']
        return image,image_name


if __name__ == '__main__':
    aa = eval_dataset("/home/yufei/HUW3/data/test_data_A/query",[],noise_type_file_path="/home/yufei/HUW4/exp/filted_extract/query_noise_type_9912.txt")
    for a,b in aa:
        #print(a.size)
        pass
    bb=aa[0]
    b=1


