'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-09-06 14:25:53
'''

import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class AMLinear(nn.Module):
    def __init__(self, in_features, num_clssses, m, s=30,weight=1.0):
        super(AMLinear,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, num_clssses))
        # self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(1))
        self.m = m
        self.n_cls = num_clssses
        self.s = s
        self.criterion = nn.CrossEntropyLoss()
        self.weight_loss = weight

    def forward(self, x, labels):
        # self.weight = nn.Parameter(F.normalize(self.weight, dim=0))
        w = self.weight
        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        x = F.normalize(x, dim=1)

        cos_theta = torch.mm(x, ww)

        cos_theta = torch.clamp(cos_theta, -1, 1)
        phi = cos_theta - self.m

        labels_one_hot = torch.zeros(len(labels), self.n_cls,device=x.device).scatter_(1, labels.unsqueeze(1), 1.)
        adjust_theta = self.s * torch.where(torch.eq(labels_one_hot, 1), phi, cos_theta)
        # adjust_theta = self.s * (cos_theta - self.m * labels_one_hot)
        if self.training:
            return self.criterion(adjust_theta,labels)*self.weight_loss
        else:
            return adjust_theta
        # return adjust_theta, cos_theta # logit,logit_margin

class AMLinearSmooth(nn.Module):
    def __init__(self, in_features, num_clssses, m, s=30,weight=1.0):
        super(AMLinearSmooth,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, num_clssses))
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(1))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.n_cls = num_clssses
        self.s = s
        self.criterion = CrossEntropyLabelSmooth(num_classes=3097,epsilon=0.1)
        self.weight_loss = weight

    def forward(self, x, labels):
        # self.weight = nn.Parameter(F.normalize(self.weight, dim=0))
        w = self.weight
        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        x = F.normalize(x, dim=1)

        cos_theta = torch.mm(x, ww)

        cos_theta = torch.clamp(cos_theta, -1, 1)
        phi = cos_theta - self.m

        labels_one_hot = torch.zeros(len(labels), self.n_cls,device=x.device).scatter_(1, labels.unsqueeze(1), 1.)
        adjust_theta = self.s * torch.where(torch.eq(labels_one_hot, 1), phi, cos_theta)
        # adjust_theta = self.s * (cos_theta - self.m * labels_one_hot)
        if self.training:
            return self.criterion(adjust_theta,labels)*self.weight_loss
        else:
            return adjust_theta

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes=3097, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size(),device=inputs.device).scatter_(1, targets.unsqueeze(1), 1)
        
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss