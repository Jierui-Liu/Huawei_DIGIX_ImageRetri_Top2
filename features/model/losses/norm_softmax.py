'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-08-29 10:52:54
'''

import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class Norm_Softmax(nn.Module):
    def __init__(self, in_features, num_clssses,s=30):
        super(Norm_Softmax,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, num_clssses))
        # self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(1))
        self.n_cls = num_clssses
        self.s = s
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x, labels):
        # self.weight = nn.Parameter(F.normalize(self.weight, dim=0))
        w = self.weight
        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        x = F.normalize(x, dim=1)
        cos_theta = torch.mm(x, ww)
        cos_theta = torch.clamp(cos_theta, -1, 1)


        # labels_one_hot = torch.zeros(len(labels), self.n_cls,device=x.device).scatter_(1, labels.unsqueeze(1), 1.)
        # adjust_theta = self.s * torch.where(torch.eq(labels_one_hot, 1), phi, cos_theta)
        adjust_theta = self.s * cos_theta
        if self.training:
            return self.criterion(adjust_theta,labels)
        else:
            return adjust_theta


