import os
from os.path import join,dirname,realpath
import pickle

def listdir(path):
    list_name=[]  
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path):  
            listdir(file_path, list_name)  
        else:  
            list_name.append(file_path)
    return list_name


root_data="/mnt/home/yufei/HWdata"
root_data_gallery=join(root_data,"test_data_A","gallery")
root_data_query=join(root_data,"test_data_A","query")
gallery_json=join('data_jsons','huawei_A_gallery.json')
query_json=join('data_jsons','huawei_A_query.json')


huawei_query={}
list_name=listdir(root_data_query)
huawei_query['nr_class']=len(list_name)
huawei_query['path_type']='absolute_path'
huawei_query['info_dicts']=[]
for img in list_name:
    dict_temp={}
    img_name=img.split('/')[-1]
    img_name=os.path.splitext(img_name)[0]

    dict_temp['path']=img
    dict_temp['label']=img_name
    dict_temp['query_name']=img_name
    huawei_query['info_dicts'].append(dict_temp)

with open(query_json, "wb") as f:
    pickle.dump(huawei_query, f)
    f.close()


huawei_gallery={}
list_name=listdir(root_data_gallery)
huawei_gallery['nr_class']=len(list_name)
huawei_gallery['path_type']='absolute_path'
huawei_gallery['info_dicts']=[]
for img in list_name:
    dict_temp={}
    img_name=img.split('/')[-1]
    img_name=os.path.splitext(img_name)[0]

    dict_temp['path']=img
    dict_temp['label']=img_name
    huawei_gallery['info_dicts'].append(dict_temp)

with open(gallery_json, "wb") as f:
    pickle.dump(huawei_gallery, f)
    f.close()