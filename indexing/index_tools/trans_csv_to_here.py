import pandas as pd
import os
from os.path import join,dirname,realpath
import pickle
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--here_dir', default=None, type=str)
    parser.add_argument('--mode', default='concat', type=str)
    parser.add_argument('--csv_dir', nargs='+')
    args = parser.parse_args()
    return args


def main():
    data_root='/mnt/home/yufei/HWdata'

    # init args
    args = parse_args()



    #query
    query_csv=[]
    for csv in args.csv_dir:
        tmp=join(csv,'query.csv')
        df=pd.read_csv(tmp,header=None)
        num_images=df.shape[0]
        num_cols=df.shape[1]
        query_csv.append(df)

    query_dir=join(args.here_dir,'query')
    if not os.path.exists(query_dir):
        os.makedirs(query_dir) 
    query_name=join(query_dir,'part_0.json')

    dist_query={}
    dist_query['nr_class']=num_images
    dist_query['path_type']='absolute_path'
    dist_query['info_dicts']=[]
    for i in range(num_images):
        dict_tmp={}
        dict_tmp['path']=join(data_root,'test_data_A',df.iloc[i,0])
        dict_tmp['label']=df.iloc[i,0]
        dict_tmp['query_name']=df.iloc[i,0]
        dict_tmp['idx']=i
        dict_tmp['feature']={}
        feature_lst=[]
        feature_array=np.zeros(num_cols-1)
        for csv in query_csv:
            if args.mode=='concat':
                feature_lst=feature_lst+list(csv.iloc[i,1:].astype(np.float32))
            elif args.mode=='add':
                feature_array=feature_array+csv.iloc[i,1:].astype(np.float32)
        if args.mode=='add':
            feature_lst=list(feature_array)
        dict_tmp['feature']['f1']=feature_lst
        dist_query['info_dicts'].append(dict_tmp)
        if(i%100==0):
            print('query:{}/{}'.format(i+1,num_images))

    with open(query_name, "wb") as f:
        pickle.dump(dist_query, f)
        
    #gallery
    gallery_csv=[]
    for csv in args.csv_dir:
        tmp=join(csv,'gallery.csv')
        df=pd.read_csv(tmp,header=None)
        num_images=df.shape[0]
        num_cols=df.shape[1]
        gallery_csv.append(df)

    gallery_dir=join(args.here_dir,'gallery')
    if not os.path.exists(gallery_dir):
        os.makedirs(gallery_dir) 
    gallery_name=join(gallery_dir,'part_0.json')

    dist_gallery={}
    dist_gallery['nr_class']=num_images
    dist_gallery['path_type']='absolute_path'
    dist_gallery['info_dicts']=[]
    for i in range(num_images):
        dict_tmp={}
        dict_tmp['path']=join(data_root,'train_data',df.iloc[i,0])
        dict_tmp['label']=df.iloc[i,0]
        dict_tmp['idx']=i
        dict_tmp['feature']={}
        feature_lst=[]
        feature_array=np.zeros(num_cols-1)
        for csv in gallery_csv:
            if args.mode=='concat':
                feature_lst=feature_lst+list(csv.iloc[i,1:].astype(np.float32))
            elif args.mode=='add':
                feature_array=feature_array+csv.iloc[i,1:].astype(np.float32)
        if args.mode=='add':
            feature_lst=list(feature_array)
        dict_tmp['feature']['f1']=feature_lst
        dist_gallery['info_dicts'].append(dict_tmp)
        if(i%100==0):
            print('gallery:{}/{}'.format(i+1,num_images))

    with open(gallery_name, "wb") as f:
        pickle.dump(dist_gallery, f)





if __name__ == '__main__':
    main()
