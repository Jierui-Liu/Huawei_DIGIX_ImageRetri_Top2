
import torch
import torch.nn as nn
import torchvision
import sys
sys.path.append("/home/yufei/HUW3")
import SRC.pre_process.opencv_transoform as cvtransform
from SRC.pre_process.attention_crop import spectral_cluster

#comment
#来自 HUW2/baseline/dense169_2048_tri_ce_448_amssoftmax_GEMP_bnhead_multigpu_eula7.py


minibatchsize_embedding=96

size=512

data_dir="/home/yufei/HUW3/data/train_data_resize{}_rgb".format(size)
attention_map_path='/home/yufei/HUW3/data/test_data_A_resize512_rgb/attention_map/query.json'
attention_map2_boxes_fn=spectral_cluster()   #用于生成 boxes的算法


pre_process_test_embedding= torchvision.transforms.Compose([
        cvtransform.RescalePad(output_size=512),
        cvtransform.ToTensor(),
        cvtransform.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=False)
    ])

