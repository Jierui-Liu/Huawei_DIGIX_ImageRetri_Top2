'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors : now more
LastEditTime: 2020-07-21 22:11:54
Description : 
'''
import data.dataset.build as datasets

def build_dataset(cfg_dataset,transforms):
    cfg_dataset = cfg_dataset.copy()
    dataset_type = cfg_dataset.pop("type")
    dataset_kwags = cfg_dataset
    
    if hasattr(datasets,dataset_type):
        dataset = getattr(datasets,dataset_type)(**dataset_kwags,transforms=transforms)
    else:
        raise ValueError("\'type\' of dataset is not defined. Got {}".format(dataset_type))
    return dataset
