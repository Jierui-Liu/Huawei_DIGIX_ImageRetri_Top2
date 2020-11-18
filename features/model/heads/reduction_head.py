'''
@Author      : now more
@Contact     : lin.honghui@qq.com
@LastEditors : now more
@Description : 
@LastEditTime: 2020-07-29 09:52:06
'''
'''
@Author      : now more
@Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
@Description : 
LastEditTime: 2020-07-28 21:22:17
'''
import torch.nn as nn
from utils import weights_init_kaiming,weights_init_classifier
import model.layers as layers

class ReductionHead(nn.Module):
    def __init__(self, in_feat,reduction_dim):
        super(ReductionHead,self).__init__()

        self.reduction_dim = reduction_dim

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_feat, reduction_dim, 1, 1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.bnneck = nn.BatchNorm2d(reduction_dim)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)


    def forward(self, features):
        global_feat = self.bottleneck(features)
        bn_feat = self.bnneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]
        # if not self.training:
        #     return bn_feat,None
        return bn_feat




