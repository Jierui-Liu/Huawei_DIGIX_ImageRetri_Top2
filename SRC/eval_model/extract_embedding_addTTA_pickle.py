'''
Author: your name
Date: 2020-08-11 05:02:51
LastEditTime: 2020-08-16 13:19:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /HW2/SRC/eval_model/extract_embedding_addTTA_pickle.py
'''

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
# def extract_embedding(model,dataloader,device):
def extract_embedding(model,dataloader):
    model.eval()
    features_list = []*len(dataloader)
    filename_list = []*len(dataloader)
    data_dict={}
    with torch.no_grad():
        for image, filename in tqdm(dataloader):

            image = image.cuda()

            features_1 = model(image.clone(), extract_embedding=True)
            features_1 = features_1.cpu().data.numpy().astype(np.float32)

            # 水平
            features_2 = model(torch.flip(image.clone(),[-1]), extract_embedding=True)
            features_2 = features_2.cpu().data.numpy().astype(np.float32)

            # 垂直
            features_3 = model(torch.flip(image.clone(),[-2]), extract_embedding=True)
            features_3 = features_3.cpu().data.numpy().astype(np.float32)
            
            # 顺时针90
            features_4 = model(torch.flip(image.clone().transpose(2,3),[-1]), extract_embedding=True)
            features_4 = features_4.cpu().data.numpy().astype(np.float32)

            # 逆时针90
            features_5 = model(torch.flip(image.clone().transpose(2,3),[-2]), extract_embedding=True)
            features_5 = features_5.cpu().data.numpy().astype(np.float32)

            features = features_1+features_2+features_3+features_4+features_5
            # import pdb; pdb.set_trace()
            features_list.append(features)
            filename_list.append(filename)
            # data_dict[filename]=list(features)
            
        np_filename = np.concatenate(filename_list)
        np_features = np.concatenate(features_list).astype(np.float32)
    data_dict["fname"]=np_filename
    data_dict["data"]=np_features



    return data_dict