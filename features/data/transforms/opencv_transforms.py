'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-09-28 10:44:07
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


class Shift_Padding(object):
    def __init__(self,p=0.1,hor_shift_ratio=0.1,ver_shift_ratio=0.1):
        assert isinstance(hor_shift_ratio,(float,list,tuple))
        if isinstance(hor_shift_ratio,float):
            self.hor_shift_ratio = (-hor_shift_ratio,hor_shift_ratio)
        else:
            assert len(hor_shift_ratio) == 2
            self.hor_shift_ratio = hor_shift_ratio

        assert isinstance(ver_shift_ratio,(float,list,tuple))
        if isinstance(ver_shift_ratio,float):
            self.ver_shift_ratio = (-ver_shift_ratio,ver_shift_ratio)
        else:
            assert len(ver_shift_ratio) == 2
            self.ver_shift_ratio = ver_shift_ratio
        self.p = p
    def __call__(self,sample):
        if np.random.random() < self.p:
            return F.shift_padding(sample,hor_shift_ratio=self.hor_shift_ratio,ver_shift_ratio=self.ver_shift_ratio)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + "(p={},hor_shift_ratio={},ver_shift_ratio={})".format(self.p,self.hor_shift_ratio,self.ver_shift_ratio)

class RandomChoice(object):
    """
    Apply transformations randomly picked from a list with a given probability
    Args:
        transforms: a list of transformations
        p: probability
    """
    def __init__(self,p,transforms):
        self.p = p
        self.transforms = transforms
    def __call__(self,sample):
        if len(self.transforms) < 1:
            raise TypeError("transforms(list) should at least have one transformation")
        for t in self.transforms:
            if np.random.uniform(0,1) < self.p:
                sample = t(sample)
        return sample

    def __repr__(self):
        return self.__class__.__name__+"(p={})".format(self.p)


class Compose(object):
    '''
    Description: Compose several transforms together
    Args (type): 
        transforms (list): list of transforms
        sample (ndarray or dict):
    return: 
        sample (ndarray or dict)
    '''
    def __init__(self,transforms):
        self.transforms = transforms
    def __call__(self,sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToTensor(object):
    '''
    Description: Convert ndarray in sample to Tensors.
    Args (type): 
        sample (ndarray or dict)
    return: 
        Converted sample.
    '''
    def __call__(self,sample):
        return F.to_tensor(sample)
    def __repr__(self):
        return self.__class__.__name__ + "()"

class Normalize(object):
    '''
    Description: Normalize a tensor with mean and standard deviation.
    Args (type): 
        mean (tuple): Sequence of means for each channel.
        std (tuple): Sequence of std for each channel.
    Return: 
        Converted sample
    '''
    def __init__(self,mean,std,inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def __call__(self,sample):
        #Convert to tensor
        mean = torch.tensor(self.mean,dtype=torch.float32)
        std = torch.tensor(self.std,dtype=torch.float32)
        return F.normalize(sample,mean,std,inplace=self.inplace)
    def __repr__(self):
        format_string = self.__class__.__name__ + "(mean={0},std={1})".format(self.mean,self.std)
        return format_string

class RandomHorizontalFlip(object):
    '''
    Description: Horizontally flip the given sample with a given probability.
    Args (type): 
        p (float): probability of the image being flipped. Default value is 0.5.
    Return: Converted sample
    '''
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sample):
        if np.random.random() < self.p:
            return F.hflip(sample)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomVerticalFlip(object):
    '''
    Description: Vertically flip the given sample with a given probability.
    Args (type): 
        p (float): probability of the image being flipped. Default value is 0.5.
    Return: 
        Converted sample
    '''
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sample):
        if np.random.random() < self.p:
            return F.vflip(sample)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)

class Lambda(object):
    '''
    Description: Apply a user-defined lambda as a transform.
    Args (type): lambd (function): Lambda/function to be used for transform.
    Return: 
    '''
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ColorJitter(object):
    '''
    Description: Randomly change the brightness, contrast and saturation of an image.
    Args (type): 
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    Return: 
        Converted sample
    '''
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        if self.saturation is not None:
            warnings.warn('Saturation jitter enabled. Will slow down loading immensely.')
        if self.hue is not None:
            warnings.warn('Hue jitter enabled. Will slow down loading immensely.')
    
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = np.random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = np.random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = np.random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = np.random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, sample):
        """
        Args:
            img (numpy ndarray): Input image.
        Returns:
            numpy ndarray: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        image = sample['image']
        if isinstance(image,np.ndarray) and image.ndim in {2,3}:
            image = transform(image)
            sample['image'] = image
            return sample
        else:
            raise TypeError("Image should be a numpu.ndarray image. Got {}".format(type(image)))

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomCrop(object):
    '''
    Description:  Crop randomly the image in a sample
    Args (type): 
        output_size(tuple or int):Desized output size.
        If int,square crop is made
    Return: 
    '''
    def __init__(self,p,output_size):
        assert isinstance(output_size,(int,tuple,list))
        if isinstance(output_size,int):
            self.output_size = (output_size,output_size)
        else:
            self.output_size = output_size
        self.p = p
    def __call__(self,sample):
        if np.random.random() < self.p:
            return F.randomcrop(sample,self.output_size)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + '(output_size={})'.format(self.output_size)

class RandomErasing(object):
    def __init__(self,p=0.5,sl=0.02,sh=0.4,rl=0.2):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.rl = rl
        self.rh = 1/rl

    def __call__(self,sample):
        if np.random.random() < self.p:
            return F.random_erasing(sample,self.sl,self.sh,self.rl,self.rh)
        else:
            return sample
    
    def __repr__(self):
        return self.__class__.__name__ + "(sl={},sh={},rl={},rh={})".format(self.sl,self.sh,self.rl,self.rh)

class Rescale(object):
    """ 
    Rescale the image in a sample to a given size.
        
        Args:
        output_size (tuple or int): Desized output size. 
            If tuple,output is matched to output_size.
            if int,smaller of image edges is matched to output_size keeping aspect ratio the same
        interpolation (int, optional): Desired interpolation. Default is ``cv2.INTER_CUBIC``
    """
    def __init__(self,output_size,interpolation=cv.INTER_LINEAR):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size
        self.interpolation = interpolation

    def __call__(self,sample):
        return F.rescale(sample,self.output_size,interpolation=self.interpolation)
    
    def __repr__(self):
        return self.__class__.__name__+'(output_size={},interpolation={})'.format(self.output_size,self.interpolation)

class RescalePad(object):
    def __init__(self,output_size,interpolation=cv.INTER_LINEAR,fill=0,padding_mode='constant'):
        self.output_size = output_size
        self.fill = fill
        self.padding_mode = padding_mode
        self.interpolation = interpolation
    def __call__(self,sample):
        sample = F.rescale_pad(sample,self.output_size,self.interpolation,self.fill,self.padding_mode)
        return sample
    def __repr__(self):
        interpolate_str = _cv2_interpolation_from_str[self.interpolation]
        return self.__class__.__name__ + '(output_size={},interpolation={},fill={},padding_mode={})'.format(self.output_size,interpolate_str,self.fill,self.padding_mode)

class RandomRotation(object):
    """
    Description: Rotate the image by  angle.
    Args:
        degree (Sequence or float or int):Range of degrees to select from.
            if degree is a number instead of sequence like (min,max),the range of degrees will be (-degrees,degrees).
        center (2-tuple,optional):Optional center of rotation. 
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.        
    """
    def __init__(self,degrees=15,center=None):

        if isinstance(degrees,numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number,it must be positive. ")
            self.degrees = (-degrees,degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence,it must be len 2.")
            self.degrees = degrees

        self.center = center

    def __call__(self,sample):

        return F.randomrotation(sample,self.degrees,center=self.center)
    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string
    

class RandomPatch(object):
    def __init__(self,p=0.5,pool_capacity=5000,min_sample_size=100,
                patch_min_area=0.01,patch_max_area=0.2,patch_min_ratio=0.2,
                p_rotate=0.5,p_flip_left_right=0.5):
        self.p = p
        self.min_sample_size = min_sample_size
        self.pool_capacity = pool_capacity
        self.patchpool = deque(maxlen=pool_capacity) # 双向队列

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio
        
        self.p_rotate = p_rotate
        self.p_flip_left_right = p_flip_left_right

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1. / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def collct_patch(self,sample):
        image = sample['image']
        if not F._is_numpy_image(image):
            raise TypeError("Image should be a numpu.ndarray image. Got {}".format(type(image)))
        
        H,W, = image.shape[:2]
        patch_w,patch_h = self.generate_wh(W,H)
        if patch_h is not None and patch_w is not None:
            x1 = random.randint(0,W-patch_w)
            y1 = random.randint(1,H-patch_h)
            patch = image[y1:y1+patch_h,x1:x1+patch_w,:]
            self.patchpool.append(patch)


        # generate patch_w,patch_h
        area = H*W
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area,self.patch_max_area)*area
            aspect_ratio = random.uniform(self.patch_min_ratio,1./self.patch_min_ratio)
            patch_h = int(round(math.sqrt(target_area * aspect_ratio)))
            patch_w = int(round(math.sqrt(target_area / aspect_ratio)))
    
    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.p_flip_left_right:
            patch = cv.flip(patch,1)
        if random.uniform(0, 1) > self.p_rotate:
            patch = F.randomrotation({'image':patch},degree=(-15,15))['image']
            pass
        return patch

    def __call__(self,sample):
        if np.random.random() < self.p:
            self.collct_patch(sample) # update patchpool
            if len(self.patchpool) < self.min_sample_size:
                return sample
            else:
                image = sample['image']
                h,w = image.shape[:2]
                patch = random.sample(self.patchpool,1)[0]
                patch_h,patch_w = patch.shape[:2]
                x1,y1 = random.randint(0,w-patch_w),random.randint(0,h-patch_h)
                patch = self.transform_patch(patch)
                image[y1:y1+patch_h,x1:x1+patch_w,:] = patch
                sample['image'] = image
                return sample

        else:
            return sample

    def __repr__(self):
        format_string = self.__class__.__name__+'(p={0},'.format(self.p)
        format_string += 'pool_capacity={},'.format(self.pool_capacity)
        format_string += 'min_sample_size={},'.format(self.min_sample_size)
        format_string += 'patch_min_area={},'.format(self.patch_min_area)
        format_string += 'patch_max_area={},'.format(self.patch_max_area)
        format_string += 'patch_min_ratio={},'.format(self.patch_min_ratio)
        format_string += 'p_rotate={},'.format(self.p_rotate)
        format_string += 'p_flip_leftright={}'.format(self.p_flip_left_right)
        format_string += ')'
        return format_string


class RandomRotate90(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sample):
        if np.random.random() < self.p:
            return F.randomrotate90(sample)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__+"(p={})".format(self.p)     

class RandomCropResized(object):
    """
    Description:
        Crop the given numpy ndarray to random size and aspect ratio.
        A crop of random size of the origin size and a random aspect 
        ratio (default: of 3/4 to 4/3) of the original aspect ratio is 
        made. This crop is finally resized to given size.

    Args:
        output_size: (int or tuple2):expected output size.
        scale: range of size of origin size cropped.
        ratio: range of aspect ratio of origin aspect ratio.
    """
    def __init__(self,p,output_size,scale,ratio):
        assert isinstance(output_size,(int,tuple,list))
        if isinstance(output_size,int):
            self.output_size = (output_size,output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        assert isinstance(scale,(tuple,list))
        assert isinstance(ratio,(tuple,list))
        self.p = p
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self,sample):
        if np.random.random() <= self.p:
            return F.randomcropresize(sample,self.p,self.output_size,self.scale,self.ratio)
        else:
            return sample
    
    def __repr__(self):
        return self.__class__.__name__ + "(p={0},output_size=({1}),scale=({2}),ratio=({3}))".format(self.p,self.output_size,self.scale,self.ratio)

class PepperNoise(object):
    def __init__(self,p=0.1,percentage=0.1):
        self.percentage = percentage
        self.p = p
    
    def __call__(self,sample):
        if np.random.random() <= self.p:
            return F.peppernoise(sample,self.percentage)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + "(p={0},percentage={1})".format(self.p,self.percentage)

class DePepperNoise(object):
    def __init__(self,p=0.1,percentage=0.1):
        self.percentage = percentage
        self.p = p
    
    def __call__(self,sample):
        if np.random.random() <= self.p:
            return F.depeppernoise(sample,self.percentage)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + "(p={0},percentage={1})".format(self.p,self.percentage)

# class GaussianNoise(object):
#     def __init__(self,percentage=0.1,p=0.1,mean=0,sigma=25):
#         self.percentage = percentage
#         self.p = p
#         self.mean = mean
#         self.sigma = sigma
    
#     def __call__(self,sample):
#         if np.random.random() <= self.p:
#             return F.gaussiannoise(sample,self.percentage,self.mean,self.sigma)
#         else:
#             return sample
#     def __repr__(self):
#         return self.__class__.__name__ + "(p={0},,percentage={1},mean={2},sigma={3})".format(self.p,self.percentage,self.mean,self.sigma)


class ChannelShuffle(object):
    def __init__(self,p=0.1):
        self.p = p
    def __call__(self,sample):
        if np.random.random() <= self.p:
            return F.channelshuffle(sample)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)

class IAAPerspective(object):
    def __init__(self,p=0.1,scale=(0.05, 0.15)):
        self.p = p
        self.scale = scale
        self.aug = A.IAAPerspective(p=p,scale=scale)
    
    def __call__(self,sample):
        image = self.aug(image=sample['image'])['image']
        sample['image'] = image
        return sample
    def __repr__(self):
        return self.__class__.__name__ + "(p={},scale={})".format(self.p,self.scale)

class ShiftScaleRotate(object):
    def __init__(self,p=0.3,shift_limit=0.1,scale_limit=(-0.5,0.2),rotate_limit=15,value=(255,255,255),border_mode=cv.BORDER_CONSTANT):
        self.p = p
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.aug = A.ShiftScaleRotate(p=p,shift_limit=shift_limit,scale_limit=scale_limit,rotate_limit=rotate_limit,value=value,border_mode=border_mode)

    def __call__(self,sample):
        image = self.aug(image=sample['image'])['image']
        sample['image'] = image
        return sample
    
    def __repr__(self):
        return self.__class__.__name__ + "(p={},shift_limit={},scale_limit={},rotate_limit={})".format(self.p,self.shift_limit,self.scale_limit,self.rotate_limit)


class MultisizePad(object):
    def __init__(self,p=0.1,resizes=[448],padsize=512):
        self.p = p
        self.resizes=resizes
        self.padsize=padsize
    def __call__(self,sample):
        if np.random.random() <= self.p:
            resize=random.sample(self.resizes,1)[0]
            return F.resize_pad(sample,resize,self.padsize)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + "(p={},resizes={},padsize={})".format(self.p,self.resizes,self.padsize)


class ImageCompression(object):
    def __init__(self,p=0.1,quality_lower=40,quality_upper=90):
        self.p = p
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        self.aug = A.ImageCompression(p=p,quality_lower=quality_lower,quality_upper=quality_upper)
        
    def __call__(self,sample):
        image =  self.aug(image=sample['image'])["image"]
        sample['image'] = image
        return sample

    def __repr__(self):
        return self.__class__.__name__ + "(p={},quality_lower={},quality_upper={})".format(self.p,self.quality_lower,self.quality_upper)


class All_Blur(object):
    def __init__(self,p=0.1,blur_limit=7):
        self.p = p
        self.blur_limit = blur_limit
        self.blur_list = [A.Blur(p=p,blur_limit=blur_limit),A.MedianBlur(p=p,blur_limit=blur_limit),A.GaussianBlur(p=p,blur_limit=blur_limit),A.MotionBlur(p=p,blur_limit=blur_limit)]
    def __call__(self,sample):
        image = np.random.choice(self.blur_list)(image=sample['image'])['image']
        sample['image'] = image
        return sample
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={},blur_limit={})".format(self.p,self.blur_limit)


        

class GaussianNoise(object):
    def __init__(self,p=0.1,loc=0,scale=(10,50)):
        self.p = p
        self.loc = loc
        self.scale = scale
        
        self.aug = A.IAAAdditiveGaussianNoise(p=p,loc=loc,scale=scale)

    def __call__(self,sample):
        image = self.aug(image=sample['image'])['image']
        sample['image'] = image
        return sample
    
    def __repr__(self):
        return self.__class__.__name__ + "(p={},loc={},scale={})".format(self.p,self.loc,self.scale)


class GridMask(object):
    def __init__(self,p=0.1,drop_ratio=0.4):
        self.p = p
        self.drop_ratio = drop_ratio
        self.aug = A.GridDropout(p=self.p,ratio=self.drop_ratio)
    
    def __call__(self,sample):
        image = self.aug(image=sample['image'])['image']
        sample['image'] = image
        return sample
    def __repr__(self):
        return self.__class__.__name__ + "(p={},drop_ratio={})".format(self.p,self.drop_ratio)


class CenterCrop(object):
    def __init__(self,drop_edge=32):
        self.drop_edge = drop_edge
    def __call__(self,sample):
        image = sample['image']
        h,w,c = image.shape
        assert (h>self.drop_edge*2) and (w>self.drop_edge*2)
        center_crop = image[self.drop_edge:h-self.drop_edge,self.drop_edge:w-self.drop_edge,:]
        sample['image'] = center_crop
        return sample
    def __repr__(self):
        return self.__class__.__name__ + "(drop_edge={})".format(self.drop_edge)

class DeGaussianNoise(object):
    def __init__(self,p=0.1,loc=0,scale=(10,50)):
        self.p = p
        self.loc = loc
        self.scale = scale
        
        self.aug = A.IAAAdditiveGaussianNoise(p=1,loc=loc,scale=scale)


    def __call__(self,sample):
        if np.random.random() <= self.p:
            image = self.aug(image=sample['image'])['image']
            image = cv.bilateralFilter(src=image, d=0, sigmaColor=100, sigmaSpace=12)
            sample['image'] = image 
        return sample
    
    def __repr__(self):
        return self.__class__.__name__ + "(p={},loc={},scale={})".format(self.p,self.loc,self.scale)

class de_bilateralFilter(object):
    def __init__(self,d=0, sigmaColor=100, sigmaSpace=12):
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace
    def __call__(self,sample):
        image,target = sample['image'],sample['target']
        if target == 2 or target==3:
            image = cv.bilateralFilter(src=image, d=self.d, sigmaColor=self.sigmaColor, sigmaSpace=self.sigmaSpace)
            sample['image'] = image
        return sample

class de_MedianBlur(object):
    def __init__(self,size=5):
            self.size = size
    def __call__(self,sample):
        image,target = sample['image'],sample['target']
        if target==1 or target==3:
            image=cv.medianBlur(image,self.size)
            sample['image'] = image
        return sample
        
         
if __name__ == "__main__":
    pass