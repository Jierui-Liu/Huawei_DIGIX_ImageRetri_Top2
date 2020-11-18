'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-08-06 13:58:28
Description : 
'''
import torch
import torch.nn as nn

class CrossEntropy(nn.Module):
    def __init__(self,in_feat=512,num_classes=3097,weight=1.0):
        super(CrossEntropy,self).__init__()
        self.linear = nn.Linear(in_feat,num_classes,bias=False)
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight
    
    def forward(self,predicts,targets=None):
        predicts = self.linear(predicts)
        if self.training:
            return self.criterion(predicts,targets)*self.weight
        else:
            return predicts