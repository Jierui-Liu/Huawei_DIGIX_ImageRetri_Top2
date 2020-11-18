'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-09-13 09:26:19
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class CosFace(nn.Module):
    def __init__(self, in_feat=1664, num_classes=3097, scale=30.0, margin=0.35,weight=1.0):
        super(CosFace, self).__init__()
        self.in_feature = in_feat
        self.out_feature = num_classes
        self.s = scale
        self.m = margin
        self.weight = Parameter(torch.Tensor(num_classes,in_feat))
        nn.init.xavier_uniform_(self.weight)
        self.weight_loss = weight
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, input, targets):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)

        output = self.s * (cosine - one_hot * self.m)
        if self.training:
            loss = self.criterion(output,targets)*self.weight_loss
            return loss
        else:
            return output

class CosFace_Dropout(nn.Module):
    def __init__(self, in_feat=1664, num_classes=3097, scale=30.0, margin=0.35,dropout_rate=0.2,weight=1.0):
        super(CosFace_Dropout, self).__init__()
        self.in_feature = in_feat
        self.out_feature = num_classes
        self.s = scale
        self.m = margin
        self.weight = Parameter(torch.Tensor(num_classes,in_feat))
        nn.init.xavier_uniform_(self.weight)
        self.weight_loss = weight
        self.criterion = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, input, targets):
        cosine = self.dropout(F.linear(F.normalize(input), F.normalize(self.weight)))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)

        output = self.s * (cosine - one_hot * self.m)
        if self.training:
            loss = self.criterion(output,targets)*self.weight_loss
            return loss
        else:
            return output




if __name__ == '__main__':
    pass