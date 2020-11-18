'''
Author: your name
Date: 2020-08-11 05:02:55
LastEditTime: 2020-08-12 21:55:58
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /HW2/SRC/Model_component/CE_loss.py
'''

import torch.nn.functional as F
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import Parameter
import math

class AMLinear_dropout(nn.Module):
    def __init__(self, in_features, n_cls, m, s=30,dropout_rate=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, n_cls))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.n_cls = n_cls
        self.s = s
        self.dropout = nn.Dropout(p=dropout_rate,inplace=True)
    def forward(self, x, labels, ):
        # self.weight = nn.Parameter(F.normalize(self.weight, dim=0))
        w = self.weight
        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        x = F.normalize(x, dim=1)

        cos_theta = self.dropout(torch.mm(x, ww))

        cos_theta = torch.clamp(cos_theta, -1, 1)
        phi = cos_theta - self.m

        labels_one_hot = torch.zeros(len(labels), self.n_cls,device=x.device).scatter_(1, labels.unsqueeze(1), 1.)
        adjust_theta = self.s * torch.where(torch.eq(labels_one_hot, 1), phi, cos_theta)
        # adjust_theta = self.s * (cos_theta - self.m * labels_one_hot)
        return adjust_theta, cos_theta

class AMLinear(nn.Module):
    def __init__(self, in_features, n_cls, m, s=30):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, n_cls))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.n_cls = n_cls
        self.s = s

    def forward(self, x, labels, ):
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
        return adjust_theta, cos_theta



class LabelCircleLossModel(nn.Module):
    def __init__(self, num_classes, m=0.35, gamma=30, feature_dim=192):
        super(LabelCircleLossModel, self).__init__()
        self.margin = m
        self.gamma = gamma
        self.weight = torch.nn.Parameter(torch.randn(feature_dim, num_classes, requires_grad=True))
        self.labels = torch.tensor([x for x in range(num_classes)]).long()
        self.classes = num_classes
        self.init_weights()
        self.O_p = 1 + self.margin
        self.O_n = -self.margin
        self.Delta_p = 1 - self.margin
        self.Delta_n = self.margin
        self.loss = nn.CrossEntropyLoss()
    def init_weights(self, pretrained=None):
        self.weight.data.normal_()

    def _forward_train(self, feat, label):
        normed_feat = torch.nn.functional.normalize(feat)
        normed_weight = torch.nn.functional.normalize(self.weight,dim=0)

        bs = label.size(0)
        tensor_tmp=self.labels.expand(bs,self.classes).to(feat.device)
        mask = label.expand(self.classes, bs).t().eq(tensor_tmp).float()
        y_true = torch.zeros((bs,self.classes),device=feat.device).scatter_(1,label.view(-1,1),1)
        y_pred = torch.mm(normed_feat,normed_weight)
        y_pred = y_pred.clamp(-1,1)
        sp = y_pred[mask == 1]
        sn = y_pred[mask == 0]

        alpha_p = (self.O_p - y_pred.detach()).clamp(min=0)
        alpha_n = (y_pred.detach() - self.O_n).clamp(min=0)

        y_pred = (y_true * (alpha_p * (y_pred - self.Delta_p)) +
                    (1-y_true) * (alpha_n * (y_pred - self.Delta_n))) * self.gamma
        loss = self.loss(y_pred,label)

        return loss, sp, sn, y_pred

    def forward(self, input, label,  mode='train'):
            if mode == 'train':
                return self._forward_train(input, label)
            elif mode == 'val':
                raise KeyError


  
class CircleLoss(nn.Module):
    def __init__(self, scale=1,margin=0.15,alpha=128):
        super(CircleLoss,self).__init__()
        self._scale = scale

        self.m = margin
        self.s = alpha

    def __call__(self, embedding, targets):
        embedding = F.normalize(embedding, dim=1)

        dist_mat = torch.matmul(embedding, embedding.t())

        N, M = dist_mat.size()
        is_pos = targets.view(N, 1).expand(N, M).eq(targets.view(M, 1).expand(M, N).t())
        is_neg = targets.view(N, 1).expand(N, M).ne(targets.view(M, 1).expand(M, N).t())

        s_p = dist_mat[is_pos].contiguous().view(N, -1)
        s_n = dist_mat[is_neg].contiguous().view(N, -1)

        alpha_p = F.relu(-s_p.detach() + 1 + self.m)
        alpha_n = F.relu(s_n.detach() + self.m)
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - self.s * alpha_p * (s_p - delta_p)
        logit_n = self.s * alpha_n * (s_n - delta_n)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss * self._scale


class CircleSoftmax(nn.Module):
    def __init__(self, in_feat,scale,margin,num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = scale
        self._m = margin
        # self.criterion = nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss()

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features, targets):
        sim_mat = F.linear(F.normalize(features), F.normalize(self.weight))
        alpha_p = F.relu(-sim_mat.detach() + 1 + self._m)
        alpha_n = F.relu(sim_mat.detach() + self._m)
        delta_p = 1 - self._m
        delta_n = self._m

        s_p = self._s * alpha_p * (sim_mat - delta_p)
        s_n = self._s * alpha_n * (sim_mat - delta_n)

        targets_ = F.one_hot(targets.clone(), num_classes=self._num_classes)

        pred_class_logits = targets_ * s_p + (1.0 - targets_) * s_n
        # print(pred_class_logits.shape,targets.shape)
        # return pred_class_logits
        return self.criterion(pred_class_logits,targets),pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )



class ArcfaceLoss_Dropout(nn.Module):
    def __init__(self, in_feat, num_classes,scale=64,margin=0.35,dropout_rate=0.3,weight=1.0):
        super(ArcfaceLoss_Dropout,self).__init__()
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
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, features, targets):
        # get cos(theta)
        cos_theta = self.dropout(F.linear(F.normalize(features), F.normalize(self.weight)))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta_clone=cos_theta.clone()
        cos_theta_clone[mask] = hard_example * (self.t + hard_example)
        cos_theta_clone.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta_clone * self._s

        loss = self.criterion(pred_class_logits,targets)

        return pred_class_logits,loss

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

        loss = self.criterion(pred_class_logits,targets)

        return pred_class_logits,loss

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )