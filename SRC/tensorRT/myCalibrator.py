# -*- coding: utf-8 -*-
"""
Created on : 20200608
@author: LWS

Create custom calibrator, use to calibrate int8 TensorRT model.

Need to override some methods of trt.IInt8EntropyCalibrator2, such as get_batch_size, get_batch,
read_calibration_cache, write_calibration_cache.

"""
import random
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import os
import numpy as np
from PIL import Image
import pandas as pd

import torchvision.transforms as transforms
import sys
import cv2 as cv
import SRC.pre_process.opencv_transoform as cvtransform


def rescale_pad_jr(image,size_new):
    h, w, c = image.shape
    if w > h:
        new_tmp=int(h * size_new * 1.0 / w) #new_tmp<size_new 宽>高
        image_tmp = cv.resize(image,(size_new, new_tmp))#宽,高
        image_new=np.zeros((size_new,size_new,3))
        start=int((size_new-new_tmp)/2)
        image_new[start:start+new_tmp,:,:]=image_tmp#高,宽,通道
    else:
        new_tmp=int(w * size_new * 1.0 / h) #new_tmp<size_new 宽<高
        image_tmp = cv.resize(image,(new_tmp, size_new)) #宽,高
        image_new=np.zeros((size_new,size_new,3))
        start=int((size_new-new_tmp)/2)
        image_new[:,start:start+new_tmp,:]=image_tmp#高,宽,通道

    image=image_new.astype(np.uint8)
    return image

class CenterNetEntropyCalibrator(trt.IInt8EntropyCalibrator2):

    # def __init__(self, args, files_path='/home/user/Downloads/datasets/train_val_files/val.txt'):
    def __init__(self, args):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = args.cache_file

        self.batch_size = args.batch_size
        self.Channel = args.channel
        self.Height = args.height
        self.Width = args.width
        self.transform = transforms.Compose([
            # cvtransform.RescalePad(output_size=self.Height),
            # transforms.Resize([self.Height, self.Width]),  # [h,w]
            transforms.ToTensor(),   
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=False)
        ])

        # self._txt_file = open(args.calib_files_path, 'r')
        # self._lines = self._txt_file.readlines()
        # np.random.shuffle(self._lines)
        # self.imgs = [os.path.join('/home/user/Downloads/datasets/train_val_files/images',
        #                           line.rstrip() + '.jpg') for line in self._lines]
        self._txt_file = pd.read_csv(args.calib_files_path, header=None)
        self._lines=list(self._txt_file[0])
        n=self.batch_size*100
        print('======num for calibration:',n)
        lst=list(random.sample(range(0,len(self._lines)),n))
        self._lines=[self._lines[l] for l in lst]
        self.imgs = [os.path.join('/home/query',
                                  line[:-4] + '.jpg') for line in self._lines]
        
        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs)//self.batch_size
        self.data_size = trt.volume([self.batch_size, self.Channel,self.Height, self.Width]) * trt.float32.itemsize
        # d_output = cuda.mem_alloc(batch_size * output.size * output.dtype.itemsize)
        self.device_input = cuda.mem_alloc(self.data_size)

    def next_batch(self):
        if self.batch_idx < self.max_batch_idx:
            batch_files = self.imgs[self.batch_idx * self.batch_size:\
                                    (self.batch_idx + 1) * self.batch_size]
            batch_imgs = np.zeros((self.batch_size, self.Channel, self.Height, self.Width),
                                  dtype=np.float32)
            for i, f in enumerate(batch_files):
                # img = Image.open(f)
                # img = np.array(img).astype(np.uint8)
                img = cv.imread(f)[...,::-1].copy()

                img=rescale_pad_jr(img,self.Height)
                img = self.transform(img).numpy()
                assert (img.nbytes == self.data_size/self.batch_size), 'not valid img!'+f
                batch_imgs[i] = img
            self.batch_idx += 1
            print("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
            return np.ascontiguousarray(batch_imgs)
        else:
            return np.array([])

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        try:
            batch_imgs = self.next_batch()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size*self.Channel*self.Height*self.Width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32))
            return [int(self.device_input)]
        except:
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
