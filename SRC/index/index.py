# -*- coding: utf-8 -*-

import os
import pickle
import time
from os.path import join,dirname,realpath

from pyretri.config import get_defaults_cfg, setup_cfg
from pyretri.index import build_index_helper, feature_loader
from pyretri.evaluate import build_evaluate_helper
import pandas as pd
import numpy as np
from shutil import copyfile 

def save_result_n(save_file,index_result_info,query_info,gallery_info):
    fp = open(save_file, 'w', encoding='utf-8')
    num=len(index_result_info)

    for i in range(num):
        lst_tmp=[]
        img_name=query_info[i]['path'].split('/')[-1][:-4]+'.jpg'
        lst_tmp.append(img_name)
        res_tmp='{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,'.format\
            (gallery_info[index_result_info[i]['ranked_neighbors_idx'][0]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][1]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][2]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][3]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][4]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][5]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][6]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][7]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][8]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][9]]['path'].split('/')[-1][:-4])
        res_tmp+='{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,'.format\
            (gallery_info[index_result_info[i]['ranked_neighbors_idx'][10]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][11]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][12]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][13]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][14]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][15]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][16]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][17]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][18]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][19]]['path'].split('/')[-1][:-4])
        res_tmp+='{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg'.format\
            (gallery_info[index_result_info[i]['ranked_neighbors_idx'][20]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][21]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][22]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][23]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][24]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][25]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][26]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][27]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][28]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][29]]['path'].split('/')[-1][:-4])
        res_tmp=img_name+',{'+res_tmp+'}'
        fp.write(res_tmp)
        fp.write('\n')
   


def index_n(config_file='/home/LinHonghui/HW2/index/index_configs/hw.yaml',\
            save_file='/home/LinHonghui/HW2/index/result/submission.csv'):
    start=time.time()
    opts=[]

    # init and load retrieval pipeline settings
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, config_file, opts)

    # load features
    query_fea, query_info, _ = feature_loader.load(cfg.index.query_fea_dir, cfg.index.feature_names)
    gallery_fea, gallery_info, _ = feature_loader.load(cfg.index.gallery_fea_dir, cfg.index.feature_names)

    print("using init_load time: {:6f}s".format(time.time()-start))
    # build helper and index features
    index_helper = build_index_helper(cfg.index)
    #gai3
    index_result_info, query_fea, gallery_fea,_ = index_helper.do_index(query_fea, query_info, gallery_fea)

    save_result_n(save_file,index_result_info,query_info,gallery_info)
    print("using total time: {:6f}s".format(time.time()-start))



def get_name_real_nopostfix(name_in):
    if name_in.split('/')[-1][-2]=='_':
        img_name_real_nopostfix_now=name_in.split('/')[-1][:-6]
    elif name_in.split('/')[-1][-2]=='_':
        img_name_real_nopostfix_now=name_in.split('/')[-1][:-7]
    else:
        img_name_real_nopostfix_now=name_in.split('/')[-1][:-4]
    return img_name_real_nopostfix_now

def save_result_segmentation(save_file,index_result_info,query_info,gallery_info,final_dist):
    fp = open(save_file, 'w', encoding='utf-8')
    num=len(index_result_info)
    query_real_nopostfix={}#query图片名：dict[gallery_name],dict[gallery_dist]

    for i in range(num):
        lst_tmp=[]
        img_name_real_nopostfix_new=get_name_real_nopostfix(query_info[i]['path'])
        # print(img_name_real_nopostfix_new,query_info[i]['path'])
        gallery_name_real_nopostfix_now=[get_name_real_nopostfix(gallery_info[img_id]['path']) for img_id in index_result_info[i]['ranked_neighbors_idx']]
        # gallery_info[index_result_info[i]['ranked_neighbors_idx'][1]]['path'].split('/')[-1][:-4],\
        gallery_dist_now=final_dist[i].tolist()
        if img_name_real_nopostfix_new in query_real_nopostfix.keys():
            query_real_nopostfix[img_name_real_nopostfix_new]['gallery_name']+=gallery_name_real_nopostfix_now
            query_real_nopostfix[img_name_real_nopostfix_new]['gallery_dist']+=gallery_dist_now
        else:
            query_real_nopostfix_now={}#dict[gallery_name],dict[gallery_dist]
            query_real_nopostfix_now['gallery_name']=gallery_name_real_nopostfix_now
            query_real_nopostfix_now['gallery_dist']=gallery_dist_now
            query_real_nopostfix[img_name_real_nopostfix_new]=query_real_nopostfix_now

    query_real_nopostfix_resort={} 
    for key in query_real_nopostfix.keys():
        print(key)
        gallery_name_real_nopostfix_now=query_real_nopostfix[key]['gallery_name']
        gallery_dist_now=query_real_nopostfix[key]['gallery_dist']
        query_real_nopostfix_now={}#gallery图片名：lst[距离]

        for index in range(len(gallery_name_real_nopostfix_now)):
            gallery_name = gallery_name_real_nopostfix_now[index]
            if gallery_name in query_real_nopostfix_now.keys():
                query_real_nopostfix_now[gallery_name].append(gallery_dist_now[index])
            else:
                query_real_nopostfix_now[gallery_name]=[]
                query_real_nopostfix_now[gallery_name].append(gallery_dist_now[index])

        gallery_dist_now_unique=[]
        gallery_name_real_nopostfix_now_unique=[]
        for key_gallery in query_real_nopostfix_now.keys():
            gallery_dist_now_unique.append(np.min(np.array(query_real_nopostfix_now[key_gallery])))
            gallery_name_real_nopostfix_now_unique.append(key_gallery)
        
        gallery_dist_now_unique=np.array(gallery_dist_now_unique)
        resorted_idx = np.argsort(gallery_dist_now_unique)
        query_real_nopostfix_resort[key]=[gallery_name_real_nopostfix_now_unique[i] for i in resorted_idx]

        #去重
        # gallery_name_real_nopostfix_unique_now = sorted(set(gallery_name_real_nopostfix_now),key=gallery_name_real_nopostfix_now.index)


    for key in query_real_nopostfix_resort:
        lst_tmp=[]
        img_name=key+'.jpg'
        lst_tmp.append(img_name)
        res_tmp='{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg'.format\
            (query_real_nopostfix_resort[key][0],\
            query_real_nopostfix_resort[key][1],\
            query_real_nopostfix_resort[key][2],\
            query_real_nopostfix_resort[key][3],\
            query_real_nopostfix_resort[key][4],\
            query_real_nopostfix_resort[key][5],\
            query_real_nopostfix_resort[key][6],\
            query_real_nopostfix_resort[key][7],\
            query_real_nopostfix_resort[key][8],\
            query_real_nopostfix_resort[key][9])

        res_tmp=img_name+',{'+res_tmp+'}'
        fp.write(res_tmp)
        fp.write('\n')

    

        

def index_segmentation(config_file='/home/LinHonghui/HW2/index/index_configs/hw.yaml',\
            save_file='/home/LinHonghui/HW2/index/result/submission.csv'):
    start=time.time()
    opts=[]

    # init and load retrieval pipeline settings
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, config_file, opts)

    # load features
    query_fea, query_info, _ = feature_loader.load(cfg.index.query_fea_dir, cfg.index.feature_names)
    gallery_fea, gallery_info, _ = feature_loader.load(cfg.index.gallery_fea_dir, cfg.index.feature_names)

    print("using init_load time: {:6f}s".format(time.time()-start))
    # build helper and index features
    index_helper = build_index_helper(cfg.index)
    index_result_info, query_fea, gallery_fea,final_dist = index_helper.do_index(query_fea, query_info, gallery_fea)

    save_result_segmentation(save_file,index_result_info,query_info,gallery_info,final_dist)
    print("using total time: {:6f}s".format(time.time()-start))


def save_result(save_file,index_result_info,query_info,gallery_info):
    fp = open(save_file, 'w', encoding='utf-8')
    num=len(index_result_info)

    for i in range(num):
        lst_tmp=[]
        img_name=query_info[i]['path'].split('/')[-1][:-4]+'.jpg'
        lst_tmp.append(img_name)
        res_tmp='{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg'.format\
            (gallery_info[index_result_info[i]['ranked_neighbors_idx'][0]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][1]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][2]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][3]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][4]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][5]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][6]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][7]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][8]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][9]]['path'].split('/')[-1][:-4])
        res_tmp=img_name+',{'+res_tmp+'}'
        fp.write(res_tmp)
        fp.write('\n')
   


def index(config_file='/home/LinHonghui/HW2/index/index_configs/hw.yaml',\
            save_file='/home/LinHonghui/HW2/index/result/submission.csv',\
            query_fea_dir=None,gallery_fea_dir=None,feature_names=None):
    start=time.time()
    opts=[]

    # init and load retrieval pipeline settings
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, config_file, opts)

    # load features
    query_fea, query_info, _ = feature_loader.load(cfg.index.query_fea_dir, cfg.index.feature_names)
    gallery_fea, gallery_info, _ = feature_loader.load(cfg.index.gallery_fea_dir, cfg.index.feature_names)

    print("using init_load time: {:6f}s".format(time.time()-start))
    # build helper and index features
    index_helper = build_index_helper(cfg.index)
    #gai3
    index_result_info, query_fea, gallery_fea,_ = index_helper.do_index(query_fea, query_info, gallery_fea)

    save_result(save_file,index_result_info,query_info,gallery_info)
    print("using total time: {:6f}s".format(time.time()-start))




def save_result_top1(save_file,index_result_info,query_info,gallery_info):
    fp = open(save_file, 'w', encoding='utf-8')
    num=len(index_result_info)

    for i in range(num):
        lst_tmp=[]
        img_name=query_info[i]['path'].split('/')[-1][:-4]+'.jpg'
        lst_tmp.append(img_name)
        res_tmp='{}.jpg, {}.jpg, {}.jpg, {}.jpg, {}.jpg, {}.jpg, {}.jpg, {}.jpg, {}.jpg, {}.jpg'.format\
            (gallery_info[index_result_info[i]['ranked_neighbors_idx'][0]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][1]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][2]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][3]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][4]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][5]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][6]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][7]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][8]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][9]]['path'].split('/')[-1][:-4])
        res_tmp=img_name+',{'+res_tmp+'}'
        fp.write(res_tmp)
        fp.write('\n')
   


def index_top1(config_file='/home/LinHonghui/HW2/index/index_configs/hw.yaml',\
            save_file='/home/LinHonghui/HW2/index/result/submission.csv'):
    start=time.time()
    opts=[]

    # init and load retrieval pipeline settings
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, config_file, opts)

    # load features
    query_fea, query_info, _ = feature_loader.load(cfg.index.query_fea_dir, cfg.index.feature_names)
    gallery_fea, gallery_info, _ = feature_loader.load(cfg.index.gallery_fea_dir, cfg.index.feature_names)

    print("using init_load time: {:6f}s".format(time.time()-start))
    # build helper and index features
    index_helper = build_index_helper(cfg.index)
    index_result_info, query_fea, gallery_fea = index_helper.do_index(query_fea, query_info, gallery_fea)

    save_result_top1(save_file,index_result_info,query_info,gallery_info)
    print("using total time: {:6f}s".format(time.time()-start))


def trans_json_to_here(here_dir='/home/LinHonghui/Datasets/Hw_ImageRetrieval/features/hw1',\
                json_dir=['/home/LinHonghui/HW2/models/baseline_1/embeddings/baseline_eula2'],\
                mode='concat'):

    data_root='/home/yufei/HUW2/data'#不重要，随便设

    #query
    query_data_json=[]
    for dict_json in json_dir:
        tmp=join(dict_json,'query.json')
        load_f=open(tmp,"rb")
        df = pickle.load(load_f)
        num_images=len(df['fname'])
        query_data_json.append(df['data'].astype(np.float32))
    num_dir=len(json_dir)
    

    query_dir=join(here_dir,'query')
    if not os.path.exists(query_dir):
        os.makedirs(query_dir) 
    query_name=join(query_dir,'part_0.json')

    dist_query={}
    dist_query['nr_class']=num_images
    dist_query['path_type']='absolute_path'
    dist_query['info_dicts']=[]
    for i in range(len(df["fname"])):
        img_name=df["fname"][i]
        dict_tmp={}
        dict_tmp['path']=join(data_root,'test_data_A',img_name)
        dict_tmp['label']=img_name
        dict_tmp['query_name']=img_name
        dict_tmp['idx']=i
        dict_tmp['feature']={}
        feature_lst_1=[[] for i in range(num_dir)]
        for j in range(num_dir):
            if mode=='concat':
                dict_tmp['feature']['f{}'.format(j+1)]=query_data_json[j][i,:].tolist()

        dist_query['info_dicts'].append(dict_tmp)
        if(i%5000==0):
            print('query:{}/{}'.format(i+1,num_images))
    
    with open(query_name, "wb") as f:
        pickle.dump(dist_query, f)
        
        
    #gallery
    gallery_data_json=[]
    for dict_json in json_dir:
        tmp=join(dict_json,'gallery.json')
        load_f=open(tmp,"rb")
        df = pickle.load(load_f)
        num_images=len(df['fname'])
        gallery_data_json.append(df['data'].astype(np.float32))
    num_dir=len(json_dir)

    gallery_dir=join(here_dir,'gallery')
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
            if mode=='concat':
                dict_tmp['feature']['f{}'.format(j+1)]=gallery_data_json[j][i,:].tolist()
                
        dist_gallery['info_dicts'].append(dict_tmp)
        if(i%5000==0):
            print('gallery:{}/{}'.format(i+1,num_images))
        i=i+1

    with open(gallery_name, "wb") as f:
        pickle.dump(dist_gallery, f)
    
    try:
        train_here_name=query_dir=join(here_dir,'train.json')
        train_ori_name=query_dir=join(json_dir[0],'train.json')
        copyfile(train_ori_name, train_here_name)
    except:
        print('====================== Something is wrong when dealing with train.json ======================')



def trans_json_to_here_add(here_dir='/home/LinHonghui/Datasets/Hw_ImageRetrieval/features/hw1',\
                json_dir=['/home/LinHonghui/HW2/models/baseline_1/embeddings/baseline_eula2'],\
                mode='concat'):

    data_root='/home/yufei/HUW2/data'#不重要，随便设

    #query
    query_data_json=[]
    for dict_json in json_dir:
        tmp=join(dict_json,'query.json')
        load_f=open(tmp,"rb")
        df = pickle.load(load_f)
        num_images=len(df['fname'])
        query_data_json.append(df['data'].astype(np.float32))
    num_dir=len(json_dir)
    

    query_dir=join(here_dir,'query')
    if not os.path.exists(query_dir):
        os.makedirs(query_dir) 
    query_name=join(query_dir,'part_0.json')

    dist_query={}
    dist_query['nr_class']=num_images
    dist_query['path_type']='absolute_path'
    dist_query['info_dicts']=[]
    for i in range(len(df["fname"])):
        img_name=df["fname"][i]
        dict_tmp={}
        dict_tmp['path']=join(data_root,'test_data_A',img_name)
        dict_tmp['label']=img_name
        dict_tmp['query_name']=img_name
        dict_tmp['idx']=i
        dict_tmp['feature']={}
        feature_lst_1=[[] for i in range(num_dir)]
        for j in range(num_dir):
            if mode=='concat':
                dict_tmp['feature']['f{}'.format(j+1)]=(query_data_json[j][i,:1664]+query_data_json[j][i,1664:]).tolist()

        dist_query['info_dicts'].append(dict_tmp)
        if(i%5000==0):
            print('query:{}/{}'.format(i+1,num_images))
    
    with open(query_name, "wb") as f:
        pickle.dump(dist_query, f)
        
        
    #gallery
    gallery_data_json=[]
    for dict_json in json_dir:
        tmp=join(dict_json,'gallery.json')
        load_f=open(tmp,"rb")
        df = pickle.load(load_f)
        num_images=len(df['fname'])
        gallery_data_json.append(df['data'].astype(np.float32))
    num_dir=len(json_dir)

    gallery_dir=join(here_dir,'gallery')
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
            if mode=='concat':
                dict_tmp['feature']['f{}'.format(j+1)]=(gallery_data_json[j][i,:1664]+gallery_data_json[j][i,1664:]).tolist()
                
        dist_gallery['info_dicts'].append(dict_tmp)
        if(i%5000==0):
            print('gallery:{}/{}'.format(i+1,num_images))
        i=i+1

    with open(gallery_name, "wb") as f:
        pickle.dump(dist_gallery, f)
    
    try:
        train_here_name=query_dir=join(here_dir,'train.json')
        train_ori_name=query_dir=join(json_dir[0],'train.json')
        copyfile(train_ori_name, train_here_name)
    except:
        print('====================== Something is wrong when dealing with train.json ======================')
