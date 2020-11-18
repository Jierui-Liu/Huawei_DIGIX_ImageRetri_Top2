'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-08-27 17:00:16
Description :
'''

import torch
import numbers
import warnings
import types
import cv2 as cv
from PIL import ImageFilter
import numpy as np
import random
from . import opencv_functional as F
from collections import deque
import math
import albumentations as A



_cv2_pad_to_str = {'constant':cv.BORDER_CONSTANT,
                   'edge':cv.BORDER_REPLICATE,
                   'reflect':cv.BORDER_REFLECT_101,
                   'symmetric':cv.BORDER_REFLECT
                  }
_cv2_interpolation_to_str= {'nearest':cv.INTER_NEAREST,
                         'bilinear':cv.INTER_LINEAR,
                         'area':cv.INTER_AREA,
                         'bicubic':cv.INTER_CUBIC,
                         'lanczos':cv.INTER_LANCZOS4}
_cv2_interpolation_from_str= {v:k for k,v in _cv2_interpolation_to_str.items()}


class Label_dependent_switcher(object):
    def __init__(self, list_of_transformer):
        self.list_of_transformer = list_of_transformer


    def __call__(self, sample):


        using_transform=self.list_of_transformer[sample["label"]]
        out_sample=using_transform(sample)
        return out_sample

class do_nothing(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        return sample


class de_GaussianNoise(object):
    def __init__(self):

        pass

    def __call__(self, sample):

        pass




class de_PepperNoise(object):
    def __init__(self):

        pass

    def __call__(self, sample):

        pass



     
class de_MeanBlur(object):
    def __init__(self,size=(5,5)):
        self.size = size
    def __call__(self,sample):
        image = sample['image']
        image=cv.blur(image,self.size)
        
        sample['image'] = image
        return sample

      
class de_MedianBlur(object):
    def __init__(self,size=5):
        self.size = size
    def __call__(self,sample):
        image = sample['image']
        image = cv.medianBlur(image,self.size)
        
        sample['image'] = image
        return sample


class de_bilateralFilter(object):
    def __init__(self,d=0, sigmaColor=100, sigmaSpace=9):
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace
    def __call__(self,sample):
        # print("debug_de_bilateralFilter")
        image = sample['image']
        image = cv.bilateralFilter(src=image, d=self.d, sigmaColor=self.sigmaColor, sigmaSpace=self.sigmaSpace)
        
        sample['image'] = image
        return sample


class de_meanshiftFilter(object):
    def __init__(self, sp=7, sr=50):
        self.sp = sp
        self.sr = sr
    def __call__(self,sample):
        image = sample['image']
        image = cv.pyrMeanShiftFiltering(src=image, sp=7, sr=50)
        
        sample['image'] = image
        return sample


if __name__ == "__main__":
    pass