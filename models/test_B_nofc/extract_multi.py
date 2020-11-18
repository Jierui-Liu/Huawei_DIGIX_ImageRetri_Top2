

import sys
sys.path.append(".")
from SRC.utils.get_gpu import get_gpu
from torch.utils.data import DataLoader
from SRC.datasets.raw_dataset_from_npy import train_dataset

from SRC.datasets.eval_dataset_with_noise import eval_dataset
import os
import torch

from SRC.eval_model.extract_embedding_addTTA_pickle import extract_embedding
# from SRC.eval_model.extract_embedding_pickle import extract_embedding
from SRC.utils.io import dict_to_file
import copy
from os.path import join,dirname,realpath


torch.multiprocessing.set_sharing_strategy('file_system')
mutationname=sys.argv[1]

mynn = __import__('mutation.' + mutationname, fromlist=True)

modeldir = sys.path[0]

temp = modeldir.split('/')
modelname = temp[-1]
best_model_savepath = modeldir + '/best_model_saved/' + mutationname+'.pth'
newest_model_savepath = modeldir + '/newest_model_saved/' + mutationname+'.pth'


embedding_dir=modeldir + '/embeddings/' + mutationname



Model=mynn.Model()


Model.load_state_dict({k.replace('module.',''):v for k, v  in torch.load(newest_model_savepath, map_location='cpu').items()})




num_of_gpu=1
gpu_list=get_gpu(num_of_gpu)
if len(gpu_list)==0:
    sys.exit(1)

Model = torch.nn.DataParallel(Model, device_ids=gpu_list).cuda()  #11S左右


gallery_ev=eval_dataset("/data_sda/LiuJierui/Dataset/HW_Retrieval/testdata_1019/gallery",mynn.pre_process_test_embedding,\
                        noise_type_file_path="")
gallery_ev_loader = DataLoader(dataset=gallery_ev, batch_size=mynn.minibatchsize_embedding*num_of_gpu, shuffle=False ,num_workers=32)
dict_gallery = extract_embedding(Model, gallery_ev_loader)
dict_gallery_nonlocal=copy.deepcopy(dict_gallery)
dict_gallery_nonlocal["data"]=dict_gallery_nonlocal["data"][:,:1664]
dict_to_file(os.path.join(embedding_dir,'nonlocal', "gallery.json"), dict_gallery_nonlocal)
dict_gallery_rag=copy.deepcopy(dict_gallery)
dict_gallery_rag["data"]=dict_gallery_rag["data"][:,1664:1664*2]
dict_to_file(os.path.join(embedding_dir,'rag', "gallery.json"), dict_gallery_rag)
dict_gallery_attention=copy.deepcopy(dict_gallery)
dict_gallery_attention["data"]=dict_gallery_attention["data"][:,1664*2:1664*3]
dict_to_file(os.path.join(embedding_dir,'attention', "gallery.json"), dict_gallery_attention)


query_ev=eval_dataset("/data_sda/LiuJierui/Dataset/HW_Retrieval/testdata_1019/query",mynn.pre_process_test_embedding,\
                        noise_type_file_path="")
query_ev_loader = DataLoader(dataset=query_ev, batch_size=mynn.minibatchsize_embedding*num_of_gpu, shuffle=False,num_workers=32)
dict_query = extract_embedding(Model, query_ev_loader)
dict_query_nonlocal=copy.deepcopy(dict_query)
dict_query_nonlocal["data"]=dict_query_nonlocal["data"][:,:1664]
dict_to_file(os.path.join(embedding_dir,'nonlocal', "query.json"), dict_query_nonlocal)
dict_query_rag=copy.deepcopy(dict_query)
dict_query_rag["data"]=dict_query_rag["data"][:,1664:1664*2]
dict_to_file(os.path.join(embedding_dir,'rag', "query.json"), dict_query_rag)
dict_query_attention=copy.deepcopy(dict_query)
dict_query_attention["data"]=dict_query_attention["data"][:,1664*2:1664*3]
dict_to_file(os.path.join(embedding_dir,'attention', "query.json"), dict_query_attention)

