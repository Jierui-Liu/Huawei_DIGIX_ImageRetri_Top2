

import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os
import numpy as np
from SRC.utils.io import file_to_dict
import cv2 as cv
from SRC.pre_process.attention_crop import spectral_cluster
def _boxes_crop(image,boxes):#注意此处的box是归一化的
    croped_images=[]
    h,w = image.shape[:2]
    for box in boxes:
        x1=int(box[1]*h)
        x2=int(box[3]*h)
        y1=int(box[0]*w)
        y2=int(box[2]*w)

        x1=max(x1,0)
        x2=min(x2,h)
        y1=max(y1,0)
        y2=min(y2,w)





        croped_images.append(image[x1:x2,y1:y2,:])
    return croped_images




class eval_dataset(Dataset):
    def __init__(self,image_dir,transforms,attention_map_path,attention_map2_boxes_fn):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_list = os.listdir(image_dir)
        self.image_list.sort()

        attention_dict = file_to_dict(attention_map_path)
        self.attention_maps = attention_dict['attention_map']
        self.attention_maps_fname = attention_dict['fname']
        self.attention_map2_boxes_fn=attention_map2_boxes_fn  #这是一个函数
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,idx):
        image_name = self.image_list[idx]
        image_name=image_name.replace(".npy", ".jpg")
        attention_map=self.attention_maps[idx]
        attention_map=np.squeeze(attention_map)
        assert self.attention_maps_fname[idx]==image_name
        boxes=self.attention_map2_boxes_fn(attention_map)
        img = np.load(os.path.join(self.image_dir,image_name)[:-3]+"npy")   #此处读取的已经是512*512，PAD过的
        images=_boxes_crop(img,boxes)
        transformed_images=[]
        images_names=[]
        for i in range(len(images)):
            images_names.append(image_name+"_"+str(i))
            img=images[i]
            sample = {'image':img}
            if self.transforms:
                sample = self.transforms(sample)
            transformed_images.append(sample['image'])
        return transformed_images,images_names


if __name__ == '__main__':
    import SRC.pre_process.opencv_transoform as cvtransform
    import torchvision
    pre_process_test_embedding = torchvision.transforms.Compose([
        cvtransform.RescalePad(output_size=512),
        cvtransform.ToTensor(),
        cvtransform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    ])

    aa = eval_dataset("/home/yufei/HUW3/data/test_data_A_resize512_rgb/query",pre_process_test_embedding,
                      '/home/yufei/HUW3/data/test_data_A_resize512_rgb/attention_map/query.json',
                      spectral_cluster())

    bb=aa[1710]


    b=1


