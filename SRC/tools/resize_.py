'''
Author: your name
Date: 2020-08-11 04:44:12
LastEditTime: 2020-08-18 02:05:09
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /HW2/index/index_tools/resize.py
'''
import os
import cv2
from os.path import join,dirname,realpath
import numpy as np
import shutil

# 保存到同个root下，test_data_A_resize320,train_data_resize320
root='/home/LinHonghui/Datasets/Hw_ImageRetrieval'

dir_names=['train_data','test_data_A']
size_new=576
cnt=0

for dir_name in dir_names:
    dir_read=join(root,dir_name)
    dir_save=join(root,dir_name+'_resize{}'.format(size_new))
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    dir_0s=os.listdir(dir_read)
    for dir_0 in dir_0s:
        if dir_0[-4:]=='.txt':
            old_name=join(dir_read,dir_0)
            new_name=join(dir_save,dir_0)
            shutil.copyfile(old_name, new_name)
            continue 

        dir_read_0=join(dir_read,dir_0)
        dir_save_0=join(dir_save,dir_0)

        if not os.path.exists(dir_save_0):
            os.makedirs(dir_save_0)
        image_names=os.listdir(dir_read_0)

        for image_name in image_names:
            image=cv2.imread(join(dir_read_0,image_name))
            h, w, c = image.shape
            if w > h:
                new_tmp=int(h * size_new * 1.0 / w) #new_tmp<size_new 宽>高
                image_tmp = cv2.resize(image,(size_new, new_tmp))#宽,高
                image_new=np.zeros((size_new,size_new,3))
                start=int((size_new-new_tmp)/2)
                image_new[start:start+new_tmp,:,:]=image_tmp#高,宽,通道
            else:
                new_tmp=int(w * size_new * 1.0 / h) #new_tmp<size_new 宽<高
                image_tmp = cv2.resize(image,(new_tmp, size_new)) #宽,高
                image_new=np.zeros((size_new,size_new,3))
                start=int((size_new-new_tmp)/2)
                image_new[:,start:start+new_tmp,:]=image_tmp#高,宽,通道

            image=image_new.astype(np.uint8)
            image_name=os.path.splitext(image_name)[0]
            np.save(join(dir_save_0,image_name+'.npy'), image)
            cnt=cnt+1
            if(cnt%100==0):
                print(cnt)




    

    
