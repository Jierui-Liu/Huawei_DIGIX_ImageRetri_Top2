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
    preds_list = []*len(dataloader)
    filename_list = []*len(dataloader)
    data_dict={}
    with torch.no_grad():
        for image, filename in tqdm(dataloader):

            image = image.cuda()

            features_1 = model(image.clone(), extract=True)
            
            pred = features_1.max(-1)[1]
            pred = pred.cpu().data.numpy().astype(np.uint8)

            # import pdb; pdb.set_trace()
            preds_list.append(pred)
            filename_list.append(filename)
            # data_dict[filename]=list(features)
            
        np_filename = np.concatenate(filename_list)
        np_preds = np.concatenate(preds_list)
    data_dict["fname"]=np_filename
    data_dict["label"]=np_preds



    return data_dict