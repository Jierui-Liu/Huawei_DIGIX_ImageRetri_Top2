
import torch
import torch.nn as nn
import torchvision
import sys
sys.path.append('.')
import SRC.pre_process.opencv_transoform as cvtransform
from SRC.During_training.utils import froze_all_param

from SRC.Model_component.CE_loss import AMLinear_dropout
import SRC.pre_process.denoise_transoform as denoise_transform
from SRC.Model_component.head import *
from SRC.Model_component.aggregator import *
from SRC.Model_component.XBM import *
from SRC.Model_component.non_local.non_local_dot_product import NONLocalBlock2D
#comment 795458


ce_tri_ratio=2
numofGPU=1
data_dir="/data_sda/LiuJierui/Dataset/HW_Retrieval/train_data"

#=====stage 1 config
stage1_images_per_classes=5
stage1_classes_per_minibatch=7
stage1_numofstep=50000
stage1_nums_addnegative=0
stage1_size=640
stage1_pre_process_train= torchvision.transforms.Compose([
        cvtransform.RescalePad(output_size=stage1_size),
        cvtransform.ShiftScaleRotate(p=0.3,shift_limit=0.1,scale_limit=(-0.5,0.2),rotate_limit=15),
        cvtransform.IAAPerspective(p=0.1,scale=(0.05, 0.15)),
        cvtransform.ChannelShuffle(p=0.1),
        cvtransform.RandomRotate90(p=0.2),
        cvtransform.RandomHorizontalFlip(p=0.5),
        cvtransform.RandomVerticalFlip(p=0.5),
        cvtransform.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
        cvtransform.ToTensor(),
        cvtransform.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=True)    #注意同步更改stage2的预处理
    ])



#=====stage 2 config
stage2_images_per_classes=5
stage2_classes_per_minibatch=7
stage2_numofstep=50000
stage2_nums_addnegative=0
stage2_size=stage1_size
stage2_pre_process_train= torchvision.transforms.Compose([
        cvtransform.RescalePad(output_size=stage2_size),
        cvtransform.ShiftScaleRotate(p=0.3,shift_limit=0.1,scale_limit=(-0.5,0.2),rotate_limit=15),
        cvtransform.IAAPerspective(p=0.1,scale=(0.05, 0.15)),
        cvtransform.ChannelShuffle(p=0.1),
        cvtransform.RandomRotate90(p=0.2),
        cvtransform.RandomHorizontalFlip(p=0.5),
        cvtransform.RandomVerticalFlip(p=0.5),
        cvtransform.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
        cvtransform.ToTensor(),
        cvtransform.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=True)
    ])



#=====extraction_config
minibatchsize_embedding=64
extraction_size=stage2_size



pre_process_test_embedding= torchvision.transforms.Compose([
        denoise_transform.Label_dependent_switcher([denoise_transform.do_nothing(),   #label为0，没有噪声
                                                    denoise_transform.de_MedianBlur(size=5), #label为1，椒盐噪声
                                                    denoise_transform.de_bilateralFilter(sigmaSpace=12)]), #label为2，高斯噪声，后面还可呀再加
        cvtransform.RescalePad(output_size=extraction_size),
        cvtransform.ToTensor(),
        cvtransform.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=False)
    ])



embeddings_dir=[
    "./models/test_B_nofc/embeddings/densenet169_nonlocal_640",
]

yaml_config="""

index:
  # path of the query set features and gallery set features.
  query_fea_dir: "$temp_dir_UNiQUE_mark/query"
  gallery_fea_dir: "$temp_dir_UNiQUE_mark/gallery"

  # name of the features to be loaded.
  # If there are multiple elements in the list, they will be concatenated on the channel-wise.
  feature_names: ["f1"]

  # a list of dimension process functions.
  dim_processors:
    names: ["Identity"]

  # function for enhancing the quality of features.
  feature_enhancer:
    name: "Identity"  # name of the feature enhancer.

  # function for calculating the distance between query features and gallery features.
  metric:
    name: "KNN"  # name of the metric.


  # function for re-ranking the results.
  re_ranker:
    name: "Fast_KReciprocal"
    Fast_KReciprocal:
      k1: 25
      k2: 6
      lambda_value: 0.5
      N: 3000
      dist_type: "euclidean_distance"
    
"""


#学习率
def getlr(epoch,step):
    if step<=stage1_numofstep:
        lr_based=0.0002
        lr=lr_based*(0.9999503585)**step
    else:
        lr_based=0.0002
        lr=lr_based*(0.9999503585)**step
    return lr
optimizer=torch.optim.Adam


# model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        n_class=3097

        backbone = torchvision.models.densenet169(pretrained=True)
        split_backbone=list(list(backbone.children())[0])

        model_forzen_part=split_backbone[:8]
        model_trainable_part=split_backbone[8:]

        froze_all_param(nn.Sequential(*model_forzen_part))
        self.backbone_before_nonlocal = nn.Sequential(*model_forzen_part,*model_trainable_part[:1])
        self.backbone_after_nonlocal = nn.Sequential(*model_trainable_part[1:])

        self.non_local = NONLocalBlock2D(1280,sub_sample=True, bn_layer=True)#这两个bool变量我随便设的
        self.backbone = nn.Sequential(self.backbone_before_nonlocal,\
                                        self.non_local,self.backbone_after_nonlocal)
        self.aggregator=GeneralizedMeanPoolingP()



    def input2embedding(self,x): #必须有，用于提取embedding
        x=self.backbone(x)

        embedding=self.aggregator(x)

        return embedding


    def forward(self,inputs,labels=None,extract_embedding=False):

        embedding = self.input2embedding(inputs)

        if extract_embedding:
            embedding = torch.flatten(embedding, 1)
            return embedding
      
        embedding = torch.flatten(embedding, 1)
        return embedding