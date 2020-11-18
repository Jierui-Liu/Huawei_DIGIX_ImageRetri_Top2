'''
@Author      : now more
@Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
@Description : 
LastEditTime: 2020-09-10 10:48:42
'''

import torch
import torch.nn as nn
from utils import weights_init_kaiming

class BNneckHead(nn.Module):
    def __init__(self,in_feat,num_classes):
        super(BNneckHead,self).__init__()
        self.bnneck = nn.BatchNorm2d(in_feat)
        self.bnneck.apply(weights_init_kaiming)
        self.bnneck.bias.requires_grad_(False)  # no shift
        
    
    def forward(self,features):
        return self.bnneck(features)[..., 0, 0]

class BNneckHead_Dropout(nn.Module):
    def __init__(self,in_feat,num_classes,dropout_rate=0.15):
        super(BNneckHead_Dropout,self).__init__()
        self.bnneck = nn.BatchNorm2d(in_feat)
        self.bnneck.apply(weights_init_kaiming)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.dropout = nn.Dropout(p=dropout_rate)
        
    
    def forward(self,features):
        return self.dropout(self.bnneck(features)[..., 0, 0])