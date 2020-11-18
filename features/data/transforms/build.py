'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors : now more
LastEditTime: 2020-07-21 22:55:06
Description : 
'''
from . import opencv_transforms as transforms

def build_transforms(cfg_transforms):
    cfg_transforms = cfg_transforms.copy()
    transforms_list = list()
    for item in cfg_transforms:
        transforms_type = item.pop("type")
        transforms_kwags = item
        if hasattr(transforms,transforms_type):
            transforms_list.append(getattr(transforms,transforms_type)(**transforms_kwags))
        else:
            raise ValueError("\'type\' of transforms is not defined. Got {}".format(transforms_type))
    print(transforms_list)
    return transforms.Compose(transforms_list)
