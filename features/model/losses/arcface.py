'''
@Author      : now more
@Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
@Description : 
LastEditTime: 2020-11-25 09:24:17
'''


import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import math
import numpy as np

class Arcface_LinearScheduler(nn.Module):
    def __init__(self,in_feat=1664,num_classes=3097,scale=35,dropout_rate=0.2,weight=1.0,start_value=0,stop_value=0.3,nr_steps=6e4):
        super(Arcface_LinearScheduler, self).__init__()
        self.loss = ArcfaceLoss_Dropout(in_feat,num_classes,scale,margin=start_value,dropout_rate=dropout_rate,weight=weight)
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))

    def forward(self, features, targets):
        self.step() # 每次迭代 更新 prob
        # print(self.loss._m)
        return self.loss(features,targets)

    def step(self):
        if self.i < len(self.drop_values):
            # import pdb; pdb.set_trace()
            self.loss._m = self.drop_values[self.i]
        self.i += 1




class ArcfaceLoss_Dropout(nn.Module):
    def __init__(self, in_feat, num_classes,scale=64,margin=0.35,dropout_rate=0.3,weight=1.0):
        super(ArcfaceLoss_Dropout,self).__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = scale
        self._m = margin
        self.weight_loss = weight



        self.weight = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.register_buffer('t', torch.zeros(1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(1))

        self.criterion = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, features, targets):
        # print(self._m)
        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m
        # get cos(theta)
        cos_theta = F.linear(self.dropout(F.normalize(features)), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        cos_theta_m = cos_theta_m.type_as(target_logit)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        # import pdb; pdb.set_trace()
        cos_theta[mask] = (hard_example * (self.t + hard_example)).type_as(target_logit)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta * self._s

        # print(pred_class_logits.shape,targets.shape)
        if self.training:
            loss = self.criterion(pred_class_logits,targets)*self.weight_loss
            return loss
        else:
            return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )
        
class ArcfaceLoss(nn.Module):
    def __init__(self, in_feat, num_classes,scale=64,margin=0.35,weight=1.0):
        super(ArcfaceLoss,self).__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = scale
        self._m = margin
        self.weight_loss = weight

        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m

        self.weight = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.register_buffer('t', torch.zeros(1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(1))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features, targets):
        # get cos(theta)
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta * self._s

        # print(pred_class_logits.shape,targets.shape)
        if self.training:
            loss = self.criterion(pred_class_logits,targets)*self.weight_loss
            return loss
        else:
            return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )
