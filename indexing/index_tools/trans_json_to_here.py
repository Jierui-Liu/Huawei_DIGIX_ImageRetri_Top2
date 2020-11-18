import pandas as pd
import os
from os.path import join,dirname,realpath
import pickle
import argparse
import numpy as np
from shutil import copyfile  


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--here_dir', default=None, type=str)
    parser.add_argument('--mode', default='concat', type=str)
    parser.add_argument('--json_dir', nargs='+')
    args = parser.parse_args()
    return args


def main():
    data_root='/home/yufei/HUW2/data'

    # init args
    args = parse_args()



    #query
    query_data_json=[]
    for dict_json in args.json_dir:
        tmp=join(dict_json,'query.json')
        # df=np.load(tmp)
        load_f=open(tmp,"rb")
        df = pickle.load(load_f)
        num_images=len(df['fname'])
        query_data_json.append(df['data'])
    query_data_json=np.array(query_data_json).astype(np.float32)#[1,9600,10240]
    # print(query_data_json.shape)
    # assert 1==0
    num_dir=len(args.json_dir)
    

    query_dir=join(args.here_dir,'query')
    if not os.path.exists(query_dir):
        os.makedirs(query_dir) 
    query_name=join(query_dir,'part_0.json')

    dist_query={}
    dist_query['nr_class']=num_images
    dist_query['path_type']='absolute_path'
    dist_query['info_dicts']=[]
    print(type(df["fname"]),query_data_json.shape)
    # assert 1==0
    for i in range(len(df["fname"])):
    # for i in range(10):
        img_name=df["fname"][i]
        dict_tmp={}
        dict_tmp['path']=join(data_root,'test_data_A',img_name)
        dict_tmp['label']=img_name
        dict_tmp['query_name']=img_name
        dict_tmp['idx']=i
        dict_tmp['feature']={}
        feature_lst_1=[]
        for j in range(num_dir):
            if args.mode=='concat':
                # print(query_data_json[j,i,:].shape,num_dir,len(df["fname"]),dict_tmp['path'])
                # assert 1==0
                feature_lst_1=feature_lst_1+query_data_json[j,i,:].tolist()
            
        dict_tmp['feature']['f1']=feature_lst_1
        dist_query['info_dicts'].append(dict_tmp)
        if(i%100==0):
            print('query:{}/{}'.format(i+1,num_images))
    
    with open(query_name, "wb") as f:
        pickle.dump(dist_query, f)
        
    #gallery
    gallery_data_json=[]
    for dict_json in args.json_dir:
        tmp=join(dict_json,'gallery.json')
        # df=np.load(tmp)
        load_f=open(tmp,"rb")
        df = pickle.load(load_f)
        num_images=len(df['fname'])
        gallery_data_json.append(df['data'])
    gallery_data_json=np.array(gallery_data_json).astype(np.float32)
    num_dir=len(args.json_dir)

    gallery_dir=join(args.here_dir,'gallery')
    if not os.path.exists(gallery_dir):
        os.makedirs(gallery_dir) 
    gallery_name=join(gallery_dir,'part_0.json')

    dist_gallery={}
    dist_gallery['nr_class']=num_images
    dist_gallery['path_type']='absolute_path'
    dist_gallery['info_dicts']=[]
    for i in range(len(df["fname"])):
        img_name=df["fname"][i]
        dict_tmp={}
        dict_tmp['path']=join(data_root,'train_data',img_name)
        dict_tmp['label']=img_name
        dict_tmp['idx']=i
        dict_tmp['feature']={}
        feature_lst_1=[]
        for j in range(num_dir):
            if args.mode=='concat':
                # print(query_data_json[j,i,:].shape,num_dir,len(df["fname"]),dict_tmp['path'])
                # assert 1==0
                feature_lst_1=feature_lst_1+gallery_data_json[j,i,:].tolist()
            
        dict_tmp['feature']['f1']=feature_lst_1
        dist_gallery['info_dicts'].append(dict_tmp)
        if(i%100==0):
            print('gallery:{}/{}'.format(i+1,num_images))
        i=i+1

    with open(gallery_name, "wb") as f:
        pickle.dump(dist_gallery, f)

    # train_here_name=query_dir=join(args.here_dir,'train.json')
    # train_ori_name=query_dir=join(args.json_dir[0],'train.json')
    # copyfile(train_ori_name, train_here_name)





if __name__ == '__main__':
    main()
