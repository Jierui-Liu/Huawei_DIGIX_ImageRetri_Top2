# -*- coding: utf-8 -*-

import torch
import numpy as np
import numbers
import torchvision.transforms.functional as F
from PIL import Image

class ResizePad(object):
    #长边resize到目标，短边pad
    def __init__(self, target_size, fill=0,padding_mode='constant',interpolation=Image.BILINEAR):

        self.target_size = target_size
        self.padding_mode = padding_mode
        self.interpolation=interpolation
        self.fill=fill


    def __call__(self, img):
        """
        Args:
            img (PIL)

        Returns:
            img (PIL)
        """
        w, h = img.size

        if w<h:
            th=self.target_size
            tw=int(self.target_size/h*w)
            img=img.resize((tw, th), self.interpolation)

            half=int((self.target_size-tw)/2)
            padding=(half,0,self.target_size-tw-half,0)

        else:
            tw=self.target_size
            th=int(self.target_size/w*h)
            img=img.resize((tw, th), self.interpolation)
            half=int((self.target_size - th) / 2)
            padding=(0,half,0,self.target_size- th-half)


        img=F.pad(img, padding, self.fill, self.padding_mode)

        return img





    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.target_size)

if __name__ == '__main__':
    #img = Image.open('/mnt/home/yufei/HWdata/train_data/DIGIX_002606/0C94AURDIJBL3MTG.jpg')
    img = Image.open('/mnt/home/yufei/HWdata/train_data/DIGIX_001432/I1VFWUCTXPQ52KA0.jpg')

    test=ResizePad(224)
    b = test(img)
    b.save('/home/yufei/HUW/debug/1.jpg')

    c=1

