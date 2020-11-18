'''
Author: your name
Date: 2020-08-18 21:36:18
LastEditTime: 2020-08-23 11:19:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /HW2/score_models/baseline/score.py
'''

import sys
sys.path.append(".")
import os
from SRC.index.index import *




mutationname =sys.argv[1]  #baseline

mynn = __import__('mutation.' + mutationname, fromlist=True)

modeldir = sys.path[0]

temp = modeldir.split('/')
modelname = temp[-1]

submission_file_path = modeldir + '/submission_file/' + mutationname+"_submission.csv"

temp_dir=modeldir + '/temp'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_csv(path):
    df = pd.read_csv(path, header=None)
    for i in range(len(df)):
        if i%1000==0:
            print(i)
        df.iloc[i, 1] = df.iloc[i, 1][1:]
        df.iloc[i, -1] = df.iloc[i, -1][:-1]
    return df.to_numpy()


print('loading file names')
print(mynn.submission_file_list)
submission_file_list=[]
for file_path in mynn.submission_file_list:
    submission_file_list.append(read_csv(file_path))
print('loaded file names')




def findTopNindex(arr, N):
    return np.argsort(arr)[::-1][:N]


def re_sort(array_list,weight_list):
    fuse_array = []
    len_array_list = len(array_list)
    len_array = len(array_list[0])
    for i in tqdm(range(len_array)):
        weight_dict = {}
        for j in range(len_array_list):
            for k in range(1, mynn.n+1):
                key = array_list[j][i][k]
                if key not in weight_dict:
                    if k<10:
                        weight_dict[key] = 1.0 / k*weight_list[j]
                    else:
                        weight_dict[key] = 1.0 / 10*weight_list[j]

                else:
                    if k<10:
                        weight_dict[key] = weight_dict[key] + 1.0 / k*weight_list[j]
                    else:
                        weight_dict[key] = weight_dict[key] + 1.0 / 10*weight_list[j]
        keys_list = list(weight_dict.keys())
        values_list = list(weight_dict.values())
        top_N_index = findTopNindex(values_list, 10)
        #         import pdb
        #         pdb.set_trace()
        #         print(len(keys_list))
        #         print(top_N_index)
        top_N_keys = ((np.array(keys_list))[top_N_index]).tolist()
        query = array_list[j][i, 0]
        top_N_keys.insert(0, query)
        fuse_array.append(top_N_keys)
    return fuse_array


fuse_array = re_sort(submission_file_list,mynn.weight_list)
print('resort finished')


def save_csv(arr, path):
    df = pd.DataFrame(arr)
    for i in range(len(df)):
        df.iloc[i, 1] = "{" + df.iloc[i, 1]
        df.iloc[i, -1] = df.iloc[i, -1] + "}"
    df.to_csv(path, header=None, index=None)


save_csv(fuse_array, submission_file_path)
print('save finished')






