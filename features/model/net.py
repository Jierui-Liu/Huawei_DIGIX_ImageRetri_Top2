'''
@Author      : now more
@Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
@Description : 
LastEditTime: 2020-11-18 16:04:53
'''

import torch
import torch.nn as nn

import model.backbone as backbones
import model.aggregation as aggregations
import model.heads as heads
import model.losses as losses
from model.layers import RGA_Module
from utils import weights_init_kaiming,weights_init_classifier
import os
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import torch.nn.functional as F
import torchvision.models as models

def build_backbone(cfg_backbone):
    backbone_type = cfg_backbone.pop('type')
    if hasattr(backbones,backbone_type):
        backbone = getattr(backbones,backbone_type)(**cfg_backbone)
        return backbone
    else:
        raise KeyError("backbone_type is invalid. Got {}".format(backbone_type))

def build_aggregation(cfg_aggregation):
    aggregation_type = cfg_aggregation.pop('type')
    if hasattr(nn,aggregation_type):
        pool_layer = getattr(nn,aggregation_type)(**cfg_aggregation)
        return pool_layer
    elif hasattr(aggregations,aggregation_type): # pool + flatten
        pool_layer = getattr(aggregations,aggregation_type)(**cfg_aggregation)
        return pool_layer
    else:
        raise KeyError("aggregation_type is invalid. Got {}".format(aggregation_type))

def build_heads(cfg_heads):
    head_type = cfg_heads.pop('type')
    if hasattr(nn,head_type):
        head = getattr(nn,head_type)(**cfg_heads)
        return head
    elif hasattr(heads,head_type):
        head = getattr(heads,head_type)(**cfg_heads)
        return head
    else:
        raise KeyError("head_type is invalid. Got {}".format(head_type))

def fix_bn(model):
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
    print("------------fix bn---------------")
    print("------------fix bn---------------")
    print("------------fix bn---------------")

def freeze_layers(model,num_layers=0): # 只处理backbone
    length_backbone = len(list(model.backbone.named_children()))
    print("len_layers : ",length_backbone)
    if length_backbone == 1: # for efficient net
        for i,(name,child) in enumerate(model.backbone.model.named_children()):
            if i < 2:
                child.eval()
                for param in child.parameters():
                    param.requires_grad = False
                print("freeze : ",name)

        # print(len(list(model.backbone.model._blocks.named_children())))
        for i,(name,child) in enumerate(model.backbone.model._blocks.named_children()): # todo
            if i <= num_layers:
                child.eval()
                for param in child.parameters():
                    param.requires_grad = False
                print("freeze : ",name)
            else:
                child.train()
                for param in child.parameters():
                    param.requires_grad = True
                print("unfreeze : ",name)
    elif length_backbone <= 2: # densenet
        for i,(name,child) in enumerate(model.backbone.named_children()): # todo 
            if i == 0:                                                   # densenet + dropblock
                for j,(name,childchild) in enumerate(child.named_children()):
                    if j < num_layers:
                        childchild.eval()
                        for param in childchild.parameters():
                            param.requires_grad = False
                        print("freeze : ",name)
                    else:
                        childchild.train()
                        for param in childchild.parameters():
                            param.requires_grad = True
                        print("unfreeze : ",name)
    else: # for resnet
        for i,(name,child) in enumerate(model.backbone.named_children()):
            if i < num_layers:
                child.eval()
                for param in child.parameters():
                    param.requires_grad = False
                print("freeze : ",name)
            else:
                child.train()
                for param in child.parameters():
                    param.requires_grad = True
                print("unfreeze : ",name)

class SBNet(nn.Module):
    def __init__(self,cfg):
        super(SBNet,self).__init__()
        self.cfg = cfg
        cfg_model = self.cfg['model']
        cfg_model = deepcopy(cfg_model)

        cfg_backbone = cfg_model['backbone']
        cfg_aggregation = cfg_model['aggregation']
        cfg_heads = cfg_model['heads']
        cfg_losses = cfg_model['losses']
        
        # for log
        # log_dir = os.path.join(cfg['log_dir'],'log_'+cfg['tag'])
        log_dir = cfg['log_dir']
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        
        self.backbone = build_backbone(cfg_backbone)
        self.pool_layer = build_aggregation(cfg_aggregation)
        self.heads = build_heads(cfg_heads)

        cfg_ce = cfg_losses[0]
        loss_type = cfg_ce.pop("type")
        self.celoss = getattr(losses,loss_type)(**cfg_ce)


        cfg_triplet = cfg_losses[1]
        loss_type = cfg_triplet.pop("type")
        assert loss_type=="TripletLoss"
        self.tripletloss = getattr(losses,"TripletLoss")(**cfg_triplet)


    
    def forward(self,inputs,targets=None,extract_features_flag=False,feature_type="after"):
        features = self.pool_layer(self.backbone(inputs))
        if extract_features_flag and (feature_type=="before"):
            return features[..., 0, 0]

        head_features = self.heads(features)
        if extract_features_flag and (feature_type=="after"):
            return head_features
        
        if extract_features_flag and (feature_type=="both"):
            return features[..., 0, 0],head_features
            
        if self.training:           #使用dp训练时，loss算完再回传，否则主卡压力较大
            # for Triplet Loss
            triplet_value = self.tripletloss(features[..., 0, 0],targets.long())
            ce_value = self.celoss(head_features,targets.long())
            self.writer.add_scalar("Triplet Loss",triplet_value.cpu().data.numpy())
            self.writer.add_scalar("CE Loss",ce_value.cpu().data.numpy())
            
            total_loss = torch.unsqueeze(triplet_value,0) + torch.unsqueeze(ce_value,0)
            return total_loss
        else:                       # inference时只计算 outputs
            loss_dict = {}
            pred_logit = self.celoss(head_features,targets.long())
            loss_dict['logit'] = pred_logit
            return loss_dict

class CLUNet(nn.Module):
    def __init__(self,cfg):
        super(CLUNet,self).__init__()
        self.cfg = cfg
        cfg_model = self.cfg['model']
        cfg_model = deepcopy(cfg_model)

        cfg_backbone = cfg_model['backbone']
        cfg_aggregation = cfg_model['aggregation']
        cfg_losses = cfg_model['losses']
        
        # for log
        log_dir = os.path.join(cfg['log_dir'],'log_'+cfg['tag'])
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        
        self.backbone = build_backbone(cfg_backbone)
        self.pool_layer = build_aggregation(cfg_aggregation)


        cfg_ce = cfg_losses
        loss_type = cfg_ce.pop("type")
        self.celoss = getattr(losses,loss_type)(**cfg_ce)
       
    def forward(self,inputs,targets=None,extract_features_flag=False,feature_type="before"):
        features = self.pool_layer(self.backbone(inputs))
        if extract_features_flag and (feature_type=="before"):
            return features[..., 0, 0]
            
        if self.training:           #使用dp训练时，loss算完再回传，否则主卡压力较大
            ce_value = self.celoss(features[..., 0, 0],targets.long())
            self.writer.add_scalar("CE Loss",ce_value.cpu().data.numpy())
            total_loss = torch.unsqueeze(ce_value,0)
            return total_loss
        else:                       # inference时只计算 outputs
            loss_dict = {}
            pred_logit = self.celoss(features[..., 0, 0],targets.long())
            loss_dict['logit'] = pred_logit
            return loss_dict






class SBNet_RGA(nn.Module):
    def __init__(self,cfg):
        super(SBNet_RGA,self).__init__()
        self.cfg = cfg
        cfg_model = self.cfg['model']
        cfg_model = deepcopy(cfg_model)

        cfg_backbone = cfg_model['backbone']
        patch = cfg_backbone['patch']
        cfg_aggregation = cfg_model['aggregation']
        cfg_heads = cfg_model['heads']
        cfg_losses = cfg_model['losses']
        
        # for log
        # log_dir = os.path.join(cfg['log_dir'],'log_'+cfg['tag'])
        log_dir = cfg['log_dir']
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        
        # backbone = build_backbone(cfg_backbone)
        backbone = models.densenet169(pretrained=True)
        split_backbone=list(list(backbone.children())[0])

        model_forzen_part=split_backbone[:8]
        model_trainable_part=split_backbone[8:]

        # froze_all_param(nn.Sequential(*model_forzen_part))
        # backbone_before_nonlocal = nn.Sequential(*model_forzen_part,*model_trainable_part[:1])
        backbone_after_nonlocal = nn.Sequential(*model_trainable_part[1:])

        my_RGA_Module=RGA_Module(1280,(patch//16)*(patch//16))

        # self.backbone = nn.Sequential(backbone_before_nonlocal,\
        #                                 my_RGA_Module,backbone_after_nonlocal)
        self.backbone = nn.Sequential(*model_forzen_part,*model_trainable_part[:1],\
                                        my_RGA_Module,backbone_after_nonlocal)
                                        
        self.pool_layer = build_aggregation(cfg_aggregation)
        self.heads = build_heads(cfg_heads)

        cfg_ce = cfg_losses[0]
        loss_type = cfg_ce.pop("type")
        self.celoss = getattr(losses,loss_type)(**cfg_ce)


        cfg_triplet = cfg_losses[1]
        loss_type = cfg_triplet.pop("type")
        assert loss_type=="TripletLoss"
        self.tripletloss = getattr(losses,"TripletLoss")(**cfg_triplet)


    
    def forward(self,inputs,targets=None,extract_features_flag=False,feature_type="after"):
        features = self.pool_layer(self.backbone(inputs))
        # if extract_features_flag and (feature_type=="before"):
        #     return features[..., 0, 0]

        head_features = self.heads(features)
        if extract_features_flag and (feature_type=="after"):
            return head_features
        
        # if extract_features_flag and (feature_type=="both"):
        #     return features[..., 0, 0],head_features
            
        if self.training:           #使用dp训练时，loss算完再回传，否则主卡压力较大
            # for Triplet Loss
            triplet_value = self.tripletloss(features[..., 0, 0],targets.long())
            ce_value = self.celoss(head_features,targets.long())
            self.writer.add_scalar("Triplet Loss",triplet_value.cpu().data.numpy())
            self.writer.add_scalar("CE Loss",ce_value.cpu().data.numpy())
            
            total_loss = torch.unsqueeze(triplet_value,0) + torch.unsqueeze(ce_value,0)
            return total_loss
        else:                       # inference时只计算 outputs
            loss_dict = {}
            pred_logit = self.celoss(head_features,targets.long())
            loss_dict['logit'] = pred_logit
            return loss_dict
        