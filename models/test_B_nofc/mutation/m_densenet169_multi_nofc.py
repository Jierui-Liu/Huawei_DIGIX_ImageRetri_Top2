
import torch
import torch.nn as nn
import torchvision
import os
import sys
sys.path.append('.')

from SRC.Model_component.aggregator import *
import models.test_B_nofc.mutation.densenet169_nonlocal_640 as densenet169_nonlocal
import models.test_B_nofc.mutation.densenet169_RAG_704 as densenet169_RAG
import models.test_B_nofc.mutation.densenet169_attention_704 as densenet169_attention
from SRC.Model_component.non_local.non_local_dot_product import NONLocalBlock2D
from SRC.Model_component.Attention import self_attention
from SRC.Model_component.RAG_module_init import RGA_Module
import SRC.pre_process.denoise_transoform as denoise_transform
import SRC.pre_process.opencv_transoform as cvtransform




ce_tri_ratio=2
numofGPU=1
#=====extraction_config
minibatchsize_embedding=32
extraction_size=704



pre_process_test_embedding= torchvision.transforms.Compose([
        denoise_transform.Label_dependent_switcher([denoise_transform.do_nothing(),   #label为0，没有噪声
                                                    denoise_transform.de_MedianBlur(size=5), #label为1，椒盐噪声
                                                    denoise_transform.de_bilateralFilter(sigmaSpace=12)]), #label为2，高斯噪声，后面还可呀再加
        cvtransform.RescalePad(output_size=extraction_size),
        cvtransform.ToTensor(),
        cvtransform.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=False)
    ])


embeddings_dir_nonlocal=[
    "./models/test_B_nofc/embeddings/m_densenet169_multi_nofc/nonlocal",
]
embeddings_dir_rag=[
    "./models/test_B_nofc/embeddings/m_densenet169_multi_nofc/rag",
]
embeddings_dir_attention=[
    "./models/test_B_nofc/embeddings/m_densenet169_multi_nofc/attention",
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

# model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        n_class=3097

        # nonlocal
        backbone_nonlocal = torchvision.models.densenet169(pretrained=True)
        split_backbone_nonlocal=list(list(backbone_nonlocal.children())[0])

        model_forzen_part=split_backbone_nonlocal[:8]
        model_trainable_part=split_backbone_nonlocal[8:]

        backbone_common = nn.Sequential(*model_forzen_part)
        backbone_before_nonlocal = nn.Sequential(*model_trainable_part[:1])
        backbone_after_nonlocal = nn.Sequential(*model_trainable_part[1:])

        my_nonlocal_Module = NONLocalBlock2D(1280,sub_sample=True, bn_layer=True)#这两个bool变量我随便设的

        self.backbone_common = backbone_common#共享
        self.nonlocal_0 = backbone_before_nonlocal
        self.nonlocal_1 = my_nonlocal_Module
        self.nonlocal_2 = backbone_after_nonlocal
        self.aggregator_nonlocal=GeneralizedMeanPoolingP()
        

        # rag
        backbone_rag = torchvision.models.densenet169(pretrained=True)
        split_backbone_rag=list(list(backbone_rag.children())[0])

        model_forzen_part=split_backbone_rag[:8]
        model_trainable_part=split_backbone_rag[8:]

        backbone_common = nn.Sequential(*model_forzen_part)
        backbone_before_rag = nn.Sequential(*model_trainable_part[:1])
        backbone_after_rag = nn.Sequential(*model_trainable_part[1:])

        my_RGA_Module=RGA_Module(1280,44*44)

        # self.backbone_common = backbone_common#共享
        self.rag_0 = backbone_before_rag
        self.rag_1 = my_RGA_Module
        self.rag_2 = backbone_after_rag
        self.aggregator_rag=GeneralizedMeanPoolingP()


        # attention
        backbone_attention = torchvision.models.densenet169(pretrained=True)
        split_backbone_attention=list(list(backbone_attention.children())[0])

        model_forzen_part=split_backbone_attention[:8]
        model_trainable_part=split_backbone_attention[8:]

        backbone_common = nn.Sequential(*model_forzen_part)
        backbone_before_attention = nn.Sequential(*model_trainable_part[:1])
        backbone_after_attention = nn.Sequential(*model_trainable_part[1:])

        my_attention_Module=self_attention(in_channels=1280)

        # self.backbone_common = backbone_common#共享
        self.attention_0 = backbone_before_attention
        self.attention_1 = my_attention_Module
        self.attention_2 = backbone_after_attention
        self.aggregator_attention=GeneralizedMeanPoolingP()





    def input2embedding(self,x): #必须有，用于提取embedding
        embedding=self.backbone_common(x)
        # print(x.shape)

        return embedding
    
    def here_nonlocal(self,embedding):
        embedding_nonlocal=self.nonlocal_0(embedding)
        embedding_nonlocal=self.nonlocal_1(embedding_nonlocal)
        embedding_nonlocal=self.nonlocal_2(embedding_nonlocal)
        embedding_nonlocal=self.aggregator_nonlocal(embedding_nonlocal)
        embedding_nonlocal = embedding_nonlocal.flatten(1)
        return embedding_nonlocal
        
    def here_nonlocal(self,embedding):
        embedding_nonlocal=self.nonlocal_0(embedding)
        embedding_nonlocal=self.nonlocal_1(embedding_nonlocal)
        embedding_nonlocal=self.nonlocal_2(embedding_nonlocal)
        embedding_nonlocal=self.aggregator_nonlocal(embedding_nonlocal)
        embedding_nonlocal = embedding_nonlocal.flatten(1)
        return embedding_nonlocal
        
    def here_rag(self,embedding):
        embedding_rag=self.rag_0(embedding)
        embedding_rag=self.rag_1(embedding_rag)
        embedding_rag=self.rag_2(embedding_rag)
        embedding_rag=self.aggregator_rag(embedding_rag)
        embedding_rag = embedding_rag.flatten(1)
        return embedding_rag

        
    def here_attention(self,embedding):
        embedding_attention=self.attention_0(embedding)
        embedding_attention,_=self.attention_1(embedding_attention)
        embedding_attention=self.attention_2(embedding_attention)
        embedding_attention=self.aggregator_attention(embedding_attention)
        embedding_attention = embedding_attention.flatten(1)
        return embedding_attention


    def forward(self,inputs,labels=None,extract_embedding=False):

        embedding = self.input2embedding(inputs)

        embedding_nonlocal=self.here_nonlocal(embedding)
        embedding_rag=self.here_rag(embedding)
        embedding_attention=self.here_attention(embedding)
        embedding_0=torch.cat((embedding_nonlocal,embedding_rag,embedding_attention),dim=1)

        # embedding1=torch.flip(embedding,[-1])
        # embedding_nonlocal=self.here_nonlocal(embedding1)
        # embedding_rag=self.here_rag(embedding1)
        # embedding_attention=self.here_attention(embedding1)
        # embedding_1=torch.cat((embedding_nonlocal,embedding_rag,embedding_attention),dim=1)

        # embedding2=torch.flip(embedding,[-2])
        # embedding_nonlocal=self.here_nonlocal(embedding2)
        # embedding_rag=self.here_rag(embedding2)
        # embedding_attention=self.here_attention(embedding2)
        # embedding_2=torch.cat((embedding_nonlocal,embedding_rag,embedding_attention),dim=1)

        # embedding3=torch.flip(embedding.transpose(2,3),[-1])
        # embedding_nonlocal=self.here_nonlocal(embedding3)
        # embedding_rag=self.here_rag(embedding3)
        # embedding_attention=self.here_attention(embedding3)
        # embedding_3=torch.cat((embedding_nonlocal,embedding_rag,embedding_attention),dim=1)
        
        # embedding4=torch.flip(embedding.transpose(2,3),[-2])
        # embedding_nonlocal=self.here_nonlocal(embedding4)
        # embedding_rag=self.here_rag(embedding4)
        # embedding_attention=self.here_attention(embedding4)
        # embedding_4=torch.cat((embedding_nonlocal,embedding_rag,embedding_attention),dim=1)


        # embedding=embedding_1+embedding_2+embedding_3+embedding_4+embedding_0
        embedding=embedding_0
        return embedding
    

if __name__ == '__main__':
    Model_nolacal=densenet169_nonlocal.Model()
    Model_RAG=densenet169_RAG.Model()
    Model_attention=densenet169_attention.Model()

    Model=Model()
    
    Model_nonlocal_state_dict=torch.load('./models/test_B_nofc/newest_model_saved/densenet169_nonlocal_640.pth', map_location='cpu')
    Model_RAG_state_dict=torch.load('./models/test_B_nofc/newest_model_saved/densenet169_RAG_704.pth', map_location='cpu')
    Model_attention_state_dict=torch.load('./models/test_B_nofc/newest_model_saved/densenet169_attention_704.pth', map_location='cpu')
    Model_state_dict=Model.state_dict()

    backbone_common_state_dict={}
    # {k.replace('backbone.0.','backbone_common.'):v for k, v  in Model_nonlocal_state_dict.items()\
    #                             if 'backbone.'in k and int(k.split('.')[2])<8}
    for k, v  in Model_nonlocal_state_dict.items():
        if 'backbone.'in k and k.split('.')[1]=='0' and (k.split('.')[2]=='0' or k.split('.')[2]=='1' or k.split('.')[2]=='2' or k.split('.')[2]=='3' or k.split('.')[2]=='4' or k.split('.')[2]=='5' or k.split('.')[2]=='6' or k.split('.')[2]=='7'):
               backbone_common_state_dict.update({k.replace('backbone.0.','backbone_common.'):v})                 
                            
    nonlocal_0_state_dict={k.replace('backbone.0.8.','nonlocal_0.0.'):v for k, v  in Model_nonlocal_state_dict.items()\
                            if 'backbone.0.8.' in k}
    nonlocal_1_state_dict={k.replace('backbone.1.','nonlocal_1.'):v for k, v  in Model_nonlocal_state_dict.items()\
                            if 'backbone.1.' in k}
    nonlocal_2_state_dict={k.replace('backbone.2.','nonlocal_2.'):v for k, v  in Model_nonlocal_state_dict.items()\
                            if 'backbone.2.' in k}
    aggregator_nonlocal_state_dict={k.replace('aggregator.','aggregator_nonlocal.'):v for k, v  in Model_nonlocal_state_dict.items()\
                            if 'aggregator.' in k}


    # normal_state_dict={k.replace('backbone.','normal.'):v for k, v  in Model_normal_state_dict.items()\
    #                         if 'backbone.'in k and int(k.split('.')[1])>=8}
    # normal_state_dict_copy=normal_state_dict.copy()
    # for key in normal_state_dict_copy.keys():
    #     num=int(key.split('.')[1])
    #     len_key=len(key.split('.')[1])
    #     key_new=key[:len(key.split('.')[0])]+'.'+str(num-8)\
    #             +key[len(key.split('.')[0]+'.'+key.split('.')[1]):]
    #     normal_state_dict.pop(key)
    #     normal_state_dict.update({key_new:normal_state_dict_copy[key]})

    # aggregator_state_dict={k:v for k, v  in Model_normal_state_dict.items()\
    #                         if 'aggregator.' in k}

                            
    rag_0_state_dict={k.replace('backbone.0.8.','rag_0.0.'):v for k, v  in Model_RAG_state_dict.items()\
                            if 'backbone.0.8.' in k}
    rag_1_state_dict={k.replace('backbone.1.','rag_1.'):v for k, v  in Model_RAG_state_dict.items()\
                            if 'backbone.1.' in k}
    rag_2_state_dict={k.replace('backbone.2.','rag_2.'):v for k, v  in Model_RAG_state_dict.items()\
                            if 'backbone.2.' in k}
    aggregator_rag_state_dict={k.replace('aggregator.','aggregator_rag.'):v for k, v  in Model_RAG_state_dict.items()\
                            if 'aggregator.' in k}
                                                        
    attention_0_state_dict={k.replace('backbone_before_attention.8.','attention_0.0.'):v for k, v  in Model_attention_state_dict.items()\
                            if 'backbone_before_attention.8.' in k}
    # print(attention_0_state_dict.)
    attention_1_state_dict={k.replace('self_attention.','attention_1.'):v for k, v  in Model_attention_state_dict.items()\
                            if 'self_attention.' in k}
    attention_2_state_dict={k.replace('backbone_after_attention.','attention_2.'):v for k, v  in Model_attention_state_dict.items()\
                            if 'backbone_after_attention.' in k}
    aggregator_attention_state_dict={k.replace('aggregator.','aggregator_attention.'):v for k, v  in Model_attention_state_dict.items()\
                            if 'aggregator.' in k}


    Model_state_dict_new={}
    Model_state_dict_new.update(backbone_common_state_dict)

    Model_state_dict_new.update(nonlocal_0_state_dict)
    Model_state_dict_new.update(nonlocal_1_state_dict)
    Model_state_dict_new.update(nonlocal_2_state_dict)
    Model_state_dict_new.update(aggregator_nonlocal_state_dict)

    Model_state_dict_new.update(rag_0_state_dict)
    Model_state_dict_new.update(rag_1_state_dict)
    Model_state_dict_new.update(rag_2_state_dict)
    Model_state_dict_new.update(aggregator_rag_state_dict)

    Model_state_dict_new.update(attention_0_state_dict)
    Model_state_dict_new.update(attention_1_state_dict)
    Model_state_dict_new.update(attention_2_state_dict)
    Model_state_dict_new.update(aggregator_attention_state_dict)

    for k in Model_state_dict_new.keys():
        if k not in Model_state_dict.keys():
            print(k)
            print('==================something wrong 1=================')
            exit(0)
    for k in Model_state_dict.keys():
        if k not in Model_state_dict_new.keys():
            print(k)
            print('==================something wrong 2=================')
            exit(0)
    torch.save(Model_state_dict_new, './models/test_B_nofc/newest_model_saved/m_densenet169_multi_nofc.pth',_use_new_zipfile_serialization=False)
    print('finished')
    


