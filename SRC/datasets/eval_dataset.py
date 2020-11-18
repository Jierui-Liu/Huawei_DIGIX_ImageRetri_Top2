

import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os


import cv2 as cv



class eval_dataset(Dataset):
    def __init__(self,image_dir,transforms):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_list = os.listdir(image_dir)
        self.image_list.sort()
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,idx):
        image_name = self.image_list[idx].replace(".npy",".jpg")
        image = cv.imread(os.path.join(self.image_dir,image_name))[...,::-1].copy()
        sample = {'image':image}
        if self.transforms:
            sample = self.transforms(sample)
        image = sample['image']
        return image,image_name


if __name__ == '__main__':
    aa = eval_dataset("/mnt/home/yufei/HWdata/test_data_A_resize320/query",[])
    for a,b in aa:
        #print(a.size)
        pass
    bb=aa[0]
    b=1


