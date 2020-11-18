'''
@Author      : now more
@Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
@Description : 
LastEditTime: 2020-09-14 04:30:01
'''

import torch
import torch.nn as nn
from utils import weights_init_kaiming

class IdentityHead(nn.Module):
    def __init__(self):
        super(IdentityHead,self).__init__()
        
    def forward(self,features):
        return features[..., 0, 0]

