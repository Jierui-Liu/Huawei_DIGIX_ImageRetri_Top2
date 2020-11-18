'''
@Author      : now more
@Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
@Description : 
LastEditTime: 2020-08-27 21:15:54
'''

import torch
import torch.nn as nn
from utils import weights_init_kaiming

class GNneckHead(nn.Module):
    def __init__(self,num_channels,num_groups=32,num_classes=3097):
        super(GNneckHead,self).__init__()
        self.gnneck = nn.GroupNorm(num_groups=num_groups,num_channels=num_channels)
        self.gnneck.apply(weights_init_kaiming)
        # self.gnneck.bias.requires_grad_(False)  # no shift
        
    
    def forward(self,features):
        return self.gnneck(features)[..., 0, 0]