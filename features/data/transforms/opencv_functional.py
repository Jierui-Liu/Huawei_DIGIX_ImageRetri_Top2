'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-09-25 13:32:29
Description : 
'''

import numpy as np
import torch
import cv2 as cv
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
import warnings
import numbers
import collections

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

def _is_tensor_image(image):
    '''
    Description:  Return whether image is torch.tensor and the number of dimensions of image.
    Reutrn : True or False.
    '''
    return torch.is_tensor(image) and image.ndimension()==3

def _is_numpy_image(image):
    '''
    Description: Return whether image is np.ndarray and the number of dimensions of image
    Return: True or False.
    '''
    return isinstance(image,np.ndarray) and (image.ndim in {2,3} )

def _is_numpy(landmarks):
    '''
    Description: Return whether landmarks is np.ndarray.
    Return: True or False
    '''
    return isinstance(landmarks,np.ndarray)


def to_tensor(sample):
    '''
    Description: Convert sample.values() to Tensor.
    Args (type): sample : {image:ndarray,target:int}
    Return: Converted sample
    '''
    # image,target = sample['image'],sample['target']
    image = sample['image']

    # _check
    if not _is_numpy_image(image):
        raise TypeError("sample should be numpy.ndarray. Got {}".format(type(image)))
        
    # handle numpy.array
    if image.ndim == 2:
        image = image[:,:,None]

    # Swap color axis because 
    # numpy image: H x W x C
    # torch image: C x H x W 
    image = torch.from_numpy(image.transpose((2,0,1)))
    if isinstance(image,torch.ByteTensor) or image.dtype == torch.uint8:
        image = image.float().div(255)
    
    sample['image'] = image
    return sample

def normalize(sample,mean,std,inplace=False):
    '''
    Description: Normalize a tensor image with mean and standard deviation.
    Args (type): 
        sample(dict) : {image:torch.Tensor,target:int}
        mean (sequnence): Sequence of means for each channel.
        std (sequence): Sequence of standard devication for each channel.
    Return: 
        Converted sample
    '''
    image = sample['image']
    if not _is_tensor_image(image):
        raise TypeError("image should be a torch image. Got {}".format(type(image)))

    if not inplace:
        image = image.clone()

    # check dtype and device 
    dtype = image.dtype
    device = image.device
    mean = torch.as_tensor(mean,dtype=dtype,device=device)
    std = torch.as_tensor(std,dtype=dtype,device=device)
    image.sub_(mean[:,None,None]).div_(std[:,None,None])
    sample['image'] = image
    return sample
        


def hflip(sample):
    '''
    Description: Horizontally flip the given sample['image']
    Args (type): 
        sample(dict) : {image:np.ndarray,target:int}
    Return: 
        Converted sample
    '''
    image = sample['image']
    if not _is_numpy_image(image):
        raise TypeError("sample['image'] should be np.ndarray image. Got {}".format(type(image)))
    if image.ndim == 2:
        image = image[:,:,None]
    image = cv.flip(image,1)
    if image.shape[2] == 1:
        image = cv.flip(image,1)[:,:,np.newaxis] #keep image.shape = H x W x 1
    else:
        image = cv.flip(image,1)
    sample['image'] = image
    return sample



    
def vflip(sample):
    '''
    Description: Vertically flip the given sample['image']
    Args (type): 
        sample(dict) : {image:np.ndarray,target:int}
    Return: 
        Converted sample
    '''
    image = sample['image']
    if not _is_numpy_image(image):
        raise TypeError("sample['image'] should be np.ndarray image. Got {}".format(type(image)))

    if image.ndim == 2:
        image = image[:,:,None]
    image = cv.flip(image,0)
    if image.shape[2] == 1:
        image = cv.flip(image,0)[:,:,np.newaxis] #keep image.shape = H x W x 1
    else:
        image = cv.flip(image,0)
    sample['image'] = image
    return sample
        
def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        numpy ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([ i*brightness_factor for i in range (0,256)]).clip(0,255).astype('uint8')
    # same thing but a bit slower
    # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    if img.shape[2]==1:
        return cv.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv.LUT(img, table)

def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an mage.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy ndarray: Contrast adjusted image.
    """
    # much faster to use the LUT construction than anything else I've tried
    # it's because you have to change dtypes multiple times
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([ (i-74)*contrast_factor+74 for i in range (0,256)]).clip(0,255).astype('uint8')
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(contrast_factor)
    if img.shape[2]==1:
        return cv.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv.LUT(img,table)

def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        numpy ndarray: Saturation adjusted image.
    """
    # ~10ms slower than PIL!
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return np.array(img)

def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        numpy ndarray: Hue adjusted image.
    """
    # After testing, found that OpenCV calculates the Hue in a call to 
    # cv2.cvtColor(..., cv2.COLOR_BGR2HSV) differently from PIL

    # This function takes 160ms! should be avoided
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    img = Image.fromarray(img)
    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return np.array(img)

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return np.array(img)

def randomcrop(sample,output_size):
    '''
    Description: randomcrop the given sample['image']
    Args (type): 
        sample(dict) : {image:np.ndarray,target:int}
    Return: 
        Converted sample
    '''
    image = sample['image']
    if not _is_numpy_image(image):
        raise TypeError("sample['image'] should be np.ndarray image. Got {}".format(type(image)))

    h,w = image.shape[:2]
    new_w,new_h = output_size

    if h<=new_h or w<=new_w: #存在图像尺寸小于crop_size的可能性
        warnings.warn("image_size ({}) is smaller than expect output_size ({}).".format((w,h),output_size))
        return sample
    else:
        top = np.random.randint(0,h-new_h)
        left = np.random.randint(0,w-new_w)
        image = image[top:top+new_h,left:left+new_w]
        sample['image'] = image
        return sample

def pad(image, padding, fill=0, padding_mode='constant'):
    """
    Description：Pad the given numpy ndarray on all sides with specified padding mode and fill value.
    Args:
        image (numpy ndarray): image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value on the edge of the image
            - reflect: pads with reflection of image (without repeating the last value on the edge)
                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image (repeating the last value on the edge)
                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]
    Returns:
        Numpy image: padded image.
    """
    if not _is_numpy_image(image):
        raise TypeError('image should be numpy ndarray. Got {}'.format(type(image)))
    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')
    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]
    if image.shape[2]==1:
        return(cv.copyMakeBorder(image, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                                 borderType=_cv2_pad_to_str[padding_mode], value=fill)[:,:,np.newaxis])
    else:
        return(cv.copyMakeBorder(image, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                                     borderType=_cv2_pad_to_str[padding_mode], value=fill))


def rescale(sample,output_size,interpolation=cv.INTER_LINEAR):
    '''
    Description: randomscale the sample['image'] to a given size.
    Args (type): 
        sample(dict) : {image:np.ndarray,target:int}
        output_size(int or tuple):
            Desized output size. If tuple,output is matched to output_size.
            if int,smaller of image edges is matched to output_size keeping aspect ratio the same
        interpolation (int, optional): Desired interpolation. Default is ``cv2.INTER_CUBIC``
    Return: 
        Converted sample
    '''
    image = sample['image']
    if not _is_numpy_image(image):
        raise TypeError("sample['image'] should be np.ndarray image. Got {}".format(type(image)))
        
    h,w = image.shape[:2]
    if isinstance(output_size,int):
        if h > w:
            # new_h,new_w = output_size*h/w,output_size
            new_h,new_w = output_size,output_size*w/h
        else:
            # new_h,new_w = output_size,output_size*w/h
            new_h,new_w = output_size*h/w,output_size
    else:
        new_h,new_w = output_size

    new_h,new_w = int(new_h),int(new_w)
    image = cv.resize(image,(new_w,new_h),interpolation=interpolation)
    sample['image'] = image
    return sample


def rescale_pad(sample,output_size,interpolation=cv.INTER_LINEAR,fill=0,padding_mode='constant'):
    sample = rescale(sample,output_size,interpolation)
    image = sample['image']
    h,w = image.shape[:2]
    padding = [(output_size-w)//2,(output_size-h)//2,output_size-w-(output_size-w)//2,output_size-h-(output_size-h)//2]
    image = pad(image,padding,fill,padding_mode)
    sample['image'] = image
    return sample

def random_erasing(sample,sl,sh,rl,rh):
    '''
    Description: Randomly selects a rectangle region in an image and erases its pixels.
    Args (type): 
        sample(dict) : {image:np.ndarray,target:int}
        sl : min erasing area.
        sh : max erasing area.
        rl : min aspect ratio.
        rh : max aspect ratio.

    Return: 
    '''
    image = sample['image']
    if not _is_numpy_image(image):
        raise TypeError("sample['image'] should be np.ndarray image. Got {}".format(type(image)))

    for attempt in range(20):
        h,w,c = image.shape
        area = h * w
        target_area = np.random.uniform(sl,sh)*area
        aspect_ratio = np.random.uniform(rl,rh)

        RE_h = int(round(np.sqrt(target_area * aspect_ratio)))
        RE_w = int(round(np.sqrt(target_area / aspect_ratio)))

        if RE_h < h and RE_w < w:
            x1 = np.random.randint(0,h-RE_h)
            y1 = np.random.randint(0,w-RE_w)
            if c == 3:
                image[x1:x1+RE_h,y1:y1+RE_w,0] = np.random.randint(0,255)
                image[x1:x1+RE_h,y1:y1+RE_w,1] = np.random.randint(0,255)
                image[x1:x1+RE_h,y1:y1+RE_w,2] = np.random.randint(0,255)
            else:
                image[x1:x1+RE_h,y1:y1+RE_w,0] = np.random.randint(0,255)
            sample['image'] = image
            return sample
    return sample




def shift_padding(sample,hor_shift_ratio,ver_shift_ratio):
    image = sample['image']
    if _is_numpy_image(image):
        raise TypeError("Image should be a numpu.ndarray image. Got {}".format(type(image)))
    
    h,w = image.shape[:2]
    new_h = h + np.int(np.round((ver_shift_ratio[1]-ver_shift_ratio[0])*h))
    new_w = w + np.int(np.round((hor_shift_ratio[1]-hor_shift_ratio[0])*w))
    if image.ndim == 2:
        new_image = np.zeros((new_h,new_w),dtype=image.dtype)
    else:
        new_image = np.zeros((new_h,new_w,image.shape[-1]),dtype=image.dtype)

    new_image[int(np.round(ver_shift_ratio[1]*h)):int(np.round(ver_shift_ratio[1]*h))+h,int(np.round(hor_shift_ratio[1]*w)):int(np.round(hor_shift_ratio[1]*w))+w] = image
    top = np.random.randint(0,int(np.round((ver_shift_ratio[1]-ver_shift_ratio[0])*h)))
    left = np.random.randint(0,int(np.round((hor_shift_ratio[1]-hor_shift_ratio[0])*w)))
    image = new_image[top:top+h,left:left+w]

    sample['image'] = image
    return sample

def randomrotation(sample,degree,center=None):
    image = sample['image']
    if not _is_numpy_image(image):
        raise TypeError("Image should be a numpu.ndarray image. Got {}".format(type(image)))

    h,w = image.shape[:2]
    angle = np.random.uniform(degree[0],degree[1])
    if center is None:
        center = (w/2,h/2)
    else:
        center = center

    M = cv.getRotationMatrix2D(center,angle,scale=1)  #M.size -->(2,3)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    new_w = int((h*sin) + (w*cos))
    new_h = int((h*cos) + (w*sin))
    M[0,2] += (new_w/2.0) - w/2.0
    M[1,2] += (new_h/2.0) - h/2.0
    image = cv.warpAffine(image,M,(new_w,new_h),borderValue=0)
    image = cv.resize(image,(w,h))

    sample['image'] = image
    return sample

def randomrotate90(sample):
    image = sample['image']
    if not _is_numpy_image(image):
        raise TypeError("Image should be a numpu.ndarray image. Got {}".format(type(image)))
    
    if np.random.random() <= 0.5: # 顺时针旋转90°
        image = cv.transpose(image)
        image = cv.flip(image,1)
    else:                         # 逆时针旋转90°
        image = cv.transpose(image)
        image = cv.flip(image,0)
    sample['image'] = image
    return sample        

def randomcropresize(sample,p,output_size,scale,ratio):
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
    image = sample['image']
    if not _is_numpy_image(image):
        raise TypeError("Image should be a numpy.ndarray image. Got {}".format(type(image)))

    h,w = image.shape[:2]
    for attempt in range(1000):
        target_area = np.random.uniform(scale[0],scale[1])*h*w
        aspect_ratio = np.random.uniform(ratio[0],ratio[1])
        new_h = int(round(np.sqrt(target_area*aspect_ratio)))
        new_w = int(round(np.sqrt(target_area/aspect_ratio)))
        if new_h >= h or new_w >= w:
            continue
        # 长宽比保持：宽 < 高， 统计结果
        if new_w > new_h:
            temp = new_w
            new_w = new_h
            new_h = temp
        top = np.random.randint(0,h-new_h)
        left = np.random.randint(0,w-new_w)
        image = image[top:top+new_h,left:left+new_w]
        image = cv.resize(image,tuple(output_size))
        sample['image'] = image
        return sample

    return sample


def peppernoise(sample,percentage=0.1):
    image = sample['image']
    if not _is_numpy_image(image):
        raise TypeError("Image should be a numpu.ndarray image. Got {}".format(type(image)))
    
    h,w = image.shape[:2]
    num_points = int(h*w*percentage)
    for i in range(num_points):
        rand_w = np.random.randint(0,w)
        rand_h = np.random.randint(0,h)
        if np.random.random()<=0.5:
            image[rand_h,rand_w] = 0
        else:
            image[rand_h,rand_w] = 255
    sample['image'] = image
    return sample

def depeppernoise(sample,percentage=0.1):
    image = sample['image']
    if not _is_numpy_image(image):
        raise TypeError("Image should be a numpu.ndarray image. Got {}".format(type(image)))
    
    h,w = image.shape[:2]
    num_points = int(h*w*percentage)
    for i in range(num_points):
        rand_w = np.random.randint(0,w)
        rand_h = np.random.randint(0,h)
        if np.random.random()<=0.5:
            image[rand_h,rand_w] = 0
        else:
            image[rand_h,rand_w] = 255
    image = cv.medianBlur(image,5)
    sample['image'] = image
    return sample


def gaussiannoise(sample,percentage=0.1,mean=0,sigma=25):
    image = sample['image']
    if not _is_numpy_image(image):
        raise TypeError("Image should be a numpu.ndarray image. Got {}".format(type(image)))
    
    h,w,c = image.shape
    num_points = int(h*w*percentage)
    for i in range(num_points):
        rand_w = np.random.randint(0,w)
        rand_h = np.random.randint(0,h)
        image[rand_h,rand_w] = image[rand_h,rand_w] + np.random.normal(mean,sigma,3)
        image[rand_h,rand_w] = np.clip(image[rand_h,rand_w],0,255)
    image = image.astype(np.uint8) # 默认np.uint8,to_tensor启动转换归一化
    sample['image'] = image
    return sample

def channelshuffle(sample):
    image = sample['image']
    if not _is_numpy_image(image):
        raise TypeError("Image should be a numpu.ndarray image. Got {}".format(type(image)))
    
    h,w,c = image.shape
    indice = np.array([0,1,2])
    np.random.shuffle(indice)
    image = image[:,:,indice]
    sample['image'] = image
    return sample


def resize_pad(sample,resize,padsize):
    image = sample['image']
    h, w, c = image.shape
    if w > h:
        new_tmp=int(h * resize * 1.0 / w) #new_tmp<size_new 宽>高
        image_tmp = cv.resize(image,(resize, new_tmp))#宽,高
        if np.random.random() <= 0.5:
            image_new=np.zeros((padsize,padsize,3))
        else:
            image_new=np.ones((padsize,padsize,3))
        start=int((padsize-new_tmp)/2)
        start_another=int((padsize-resize)/2)
        image_new[start:start+new_tmp,start_another:start_another+resize,:]=image_tmp#高,宽,通道
    else:
        new_tmp=int(w * resize * 1.0 / h) #new_tmp<size_new 宽<高
        image_tmp = cv.resize(image,(new_tmp, resize)) #宽,高
        if np.random.random() <= 0.5:
            image_new=np.zeros((padsize,padsize,3))
        else:
            image_new=np.ones((padsize,padsize,3))
        start=int((padsize-new_tmp)/2)
        start_another=int((padsize-resize)/2)
        image_new[start_another:start_another+resize,start:start+new_tmp,:]=image_tmp#高,宽,通道
    
    sample['image'] = image_new.astype(np.uint8)
    return sample