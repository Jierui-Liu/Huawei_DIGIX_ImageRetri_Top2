'''
Author: your name
Date: 2020-08-11 04:44:12
LastEditTime: 2020-08-22 21:10:36
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /HW2/index/index_tools/resize.py
'''
import sys
sys.path.append("/home/yufei/HUW3")
import os
import cv2
from os.path import join,dirname,realpath
import numpy as np
import shutil
import random
from torchvision import transforms
import SRC.pre_process.opencv_transoform as cvtransform
# 保存到同个root下，test_data_A_resize320,train_data_resize320


root='/home/yufei/HUW3/data'

dir_names=['test_data_A']

cnt=0


add_Pepper=cvtransform.PepperNoise(p=1.0)
add_Gaussian=cvtransform.GaussianNoise_correction(p=1.0,mean=0,sigma=25)


label_file=open("/home/yufei/HUW3/data/test_data_A_with_noise/noise_type.txt","w")



for dir_name in dir_names:
    dir_read=join(root,dir_name)
    dir_save=join(root,dir_name+'_with_noise')
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    dir_0s=os.listdir(dir_read)
    dir_0s.reverse()
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

            s_imag_path=join(dir_read_0, image_name)
            d_imag_path=join(dir_save_0, image_name)
            sr=random.random()

            if sr>=0.2: #不加噪声，直接sylink
                os.symlink(s_imag_path, d_imag_path)
                noise_type=0
            else:#加噪声
                image = cv2.imread(s_imag_path)[..., ::-1]
                image = image.astype(np.uint8)
                sample = {"image": image}
                if sr<=0.1: #加pepper噪声，注意这个transform概率是1
                    sample=add_Pepper(sample)
                    noise_type = 1
                else:
                    sample=add_Gaussian(sample)
                    noise_type = 2
                img = transforms.ToPILImage()(sample["image"])
                img.save(d_imag_path)
            if "gallery" in dir_read_0:
                label_file.writelines(image_name+","+str(noise_type)+"\n")
                #SDFGGDAAW.jpg,0  #无
                #FADFASDFASOFAW.jpg,1   #椒盐
                #SDGASDGASDGDS.jpg,2   #高斯  #都在gallery



            cnt=cnt+1
            if(cnt%100==0):
                print(cnt)


label_file.close()

    

    
