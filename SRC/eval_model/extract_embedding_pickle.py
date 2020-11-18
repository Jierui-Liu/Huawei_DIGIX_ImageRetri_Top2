
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
def extract_embedding(model,dataloader):
    model.eval()
    features_list = []*len(dataloader)
    filename_list = []*len(dataloader)
    with torch.no_grad():
        for image, filename in tqdm(dataloader):
            image = image.cuda()
            features = model(image, extract_embedding=True)
            features = features.cpu().data.numpy().astype(np.float32)
            features_list.append(features)
            filename_list.append(filename)
            

            
        np_filename = np.concatenate(filename_list)
        np_features = np.concatenate(features_list).astype(np.float32)
    data_dict={}
    data_dict["fname"]=np_filename
    data_dict["data"]=np_features



    return data_dict