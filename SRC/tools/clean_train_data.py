

import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os

from PIL import Image

if __name__ == '__main__':
    data_dir="/mnt/home/yufei/HWdata/train_data"
    labels = pd.read_csv("/mnt/home/yufei/HWdata/train_data/label.txt", header=None)

    print(len(labels))
    for i in range(len(labels)):
        label = labels[1][i]  # 根据索引index获取该图片
        img_path = os.path.join(data_dir, labels[0][i])  # 获取索引为index的图片的路径名
        corrupted=False
        try:
            img = Image.open(img_path)
            if len(img.split())!=3:
                corrupted=True
        except:
            corrupted=True


        if corrupted==True:
            labels = labels.drop(axis=0, index=i)

    print(len(labels))
    labels=labels.reset_index(drop=True)
    labels.to_csv("/mnt/home/yufei/HWdata/train_data/label_filltered.txt",header=False,index=False)