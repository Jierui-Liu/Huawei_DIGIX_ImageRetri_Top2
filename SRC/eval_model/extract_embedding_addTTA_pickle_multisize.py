'''
Author: your name
Date: 2020-08-11 05:02:51
LastEditTime: 2020-08-20 08:26:46
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
    # model_small=torch.nn.functional.interpolate(input, size=(384,384), mode='bilinear').cuda()
    # model_large=torch.nn.functional.interpolate(input, size=(512,512), mode='bilinear').cuda()

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

            ################################################################
            image_small=torch.nn.functional.interpolate(image, size=(384,384), mode='bilinear').cuda()
            features_s1 = model(image_small.clone(), extract_embedding=True)
            features_s1 = features_s1.cpu().data.numpy().astype(np.float32)

            # 水平
            features_s2 = model(torch.flip(image_small.clone(),[-1]), extract_embedding=True)
            features_s2 = features_s2.cpu().data.numpy().astype(np.float32)

            # 垂直
            features_s3 = model(torch.flip(image_small.clone(),[-2]), extract_embedding=True)
            features_s3 = features_s3.cpu().data.numpy().astype(np.float32)
            
            # 顺时针90
            features_s4 = model(torch.flip(image_small.clone().transpose(2,3),[-1]), extract_embedding=True)
            features_s4 = features_s4.cpu().data.numpy().astype(np.float32)

            # 逆时针90
            features_s5 = model(torch.flip(image_small.clone().transpose(2,3),[-2]), extract_embedding=True)
            features_s5 = features_s5.cpu().data.numpy().astype(np.float32)

            # features_1=np.concatenate((features_s1,features_1),axis=1)
            # features_2=np.concatenate((features_s2,features_2),axis=1)
            # features_3=np.concatenate((features_s3,features_3),axis=1)
            # features_4=np.concatenate((features_s4,features_4),axis=1)
            # features_5=np.concatenate((features_s5,features_5),axis=1)

            ################################################################
            image_large=torch.nn.functional.interpolate(image, size=(512,512), mode='bilinear').cuda()
            features_l1 = model(image_large.clone(), extract_embedding=True)
            features_l1 = features_l1.cpu().data.numpy().astype(np.float32)

            # 水平
            features_l2 = model(torch.flip(image_large.clone(),[-1]), extract_embedding=True)
            features_l2 = features_l2.cpu().data.numpy().astype(np.float32)

            # 垂直
            features_l3 = model(torch.flip(image_large.clone(),[-2]), extract_embedding=True)
            features_l3 = features_l3.cpu().data.numpy().astype(np.float32)
            
            # 顺时针90
            features_l4 = model(torch.flip(image_large.clone().transpose(2,3),[-1]), extract_embedding=True)
            features_l4 = features_l4.cpu().data.numpy().astype(np.float32)

            # 逆时针90
            features_l5 = model(torch.flip(image_large.clone().transpose(2,3),[-2]), extract_embedding=True)
            features_l5 = features_l5.cpu().data.numpy().astype(np.float32)


            features_1=np.concatenate((features_s1,features_1,features_l1),axis=1)
            features_2=np.concatenate((features_s2,features_2,features_l2),axis=1)
            features_3=np.concatenate((features_s3,features_3,features_l3),axis=1)
            features_4=np.concatenate((features_s4,features_4,features_l4),axis=1)
            features_5=np.concatenate((features_s5,features_5,features_l5),axis=1)

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