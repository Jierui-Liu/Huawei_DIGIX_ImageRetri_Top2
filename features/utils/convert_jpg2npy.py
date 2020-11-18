'''
@Author      : now more
@Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
@Description : 
LastEditTime: 2020-11-05 12:20:01
'''
import os
import cv2
from os.path import join,dirname,realpath
import numpy as np
import shutil
from argparse import ArgumentParser

def load_arg():
    parser = ArgumentParser(description="Convert JPG file to NPY for acceleration")
    parser.add_argument("-root_dir",type=str,required=True)
    parser.add_argument("-save_dir",type=str,required=True)
    parser.add_argument("-patch",type=int,default=512)
    arg = parser.parse_args()
    return arg



if __name__ == "__main__":
    arg = load_arg()

    # 保存到同个root下，test_data_A_resize320,train_data_resize320
    # root='/home/LinHonghui/Datasets/HW_ImageRetrieval/'
    root = arg.root_dir
    size_new = arg.patch
    save_dir = arg.save_dir

    dir_names=['test_data_A','train_data'] 
    cnt=0
    for dir_name in dir_names:
        dir_read=join(root,dir_name)
        dir_save=join(save_dir,dir_name+'_resize{}'.format(size_new))
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
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                h, w, c = image.shape
                if w > h:
                    new_tmp=int(h * size_new * 1.0 / w)
                    image_tmp = cv2.resize(image,(size_new, new_tmp))
                    image_new=np.zeros((size_new,size_new,3))
                    # image_new[:,:] = np.array([124,116,104]) # imagenet mean
                    start=int((size_new-new_tmp)/2)
                    image_new[start:start+new_tmp,:,:]=image_tmp
                else:
                    new_tmp=int(w * size_new * 1.0 / h)
                    image_tmp = cv2.resize(image,(new_tmp, size_new))
                    image_new=np.zeros((size_new,size_new,3))
                    # image_new[:,:] = np.array([124,116,104]) # imagenet mean
                    start=int((size_new-new_tmp)/2)
                    image_new[:,start:start+new_tmp,:]=image_tmp

                image=image_new.astype(np.uint8)
                image_name=os.path.splitext(image_name)[0]
                np.save(join(dir_save_0,image_name+'.npy'), image)
                cnt=cnt+1
                if(cnt%100==0):
                    print(cnt)




    

    
