'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-09-15 03:30:29
Description : 
'''
import torch
import torch.nn as nn
import torchvision.models.densenet as densenet
import torch.nn.functional as F

from model.layers import *
from model.layers.dropblock import LinearScheduler,DropBlock2D

class densenet169_dropout2d(nn.Module):
    def __init__(self,pretrained=True, progress=True,
                cfg_dropout2d=dict(p=0.1),
                **kwargs):
        super(densenet169_dropout2d,self).__init__()
        backbone = densenet.densenet169(pretrained,progress,**kwargs)
        self.backbone = backbone.features
        self.dropout2d = nn.Dropout2d(**cfg_dropout2d)

    def forward(self,x):
        out = self.dropout2d(self.backbone(x))
        out = F.relu(out, inplace=True)
        return out # 1664 channel



class densenet121(nn.Module):  
    def __init__(self,pretrained=True, progress=True, **kwargs):
        super(densenet121,self).__init__()
        backbone = densenet.densenet121(pretrained,progress,**kwargs)
        self.backbone = backbone.features
    def forward(self,x):
        out = F.relu(self.backbone(x), inplace=True)
        return out # 1024 channel

class densenet169(nn.Module):
    def __init__(self,pretrained=True, progress=True, **kwargs):
        super(densenet169,self).__init__()
        backbone = densenet.densenet169(pretrained,progress,**kwargs)
        self.backbone = backbone.features
    def forward(self,x):
        out = F.relu(self.backbone(x), inplace=True)
        return out # 1664 channel

# class densenet169_dropblock(nn.Module):
#     def __init__(self,cfg_dropblock=dict(keep_prob=0.9,block_size=7),
#                         cfg_lr=dict(start_value=1,stop_value=0.9,nr_steps=60000),
#                         pretrained=True, progress=True, **kwargs):
#         super(densenet169_dropblock,self).__init__()
#         backbone = densenet.densenet169(pretrained,progress,**kwargs)
#         self.backbone = backbone.features
#         dropblock = DropBlock2D(**cfg_dropblock)
#         self.lr = LinearScheduler(dropblock,**cfg_lr)
#     def forward(self,x):
#         out = F.relu(self.backbone(x), inplace=True)
#         out = self.lr(out)
#         return out # 1664 channel

class densenet201(nn.Module):
    def __init__(self,pretrained=True, progress=True, **kwargs):
        super(densenet201,self).__init__()
        backbone = densenet.densenet201(pretrained,progress,**kwargs)
        self.backbone = backbone.features
    def forward(self,x):
        out = F.relu(self.backbone(x), inplace=True)
        return out # 1920 channel

class densenet161(nn.Module):
    def __init__(self,pretrained=True, progress=True, **kwargs):
        super(densenet161,self).__init__()
        backbone = densenet.densenet161(pretrained,progress,**kwargs)
        self.backbone = backbone.features
    def forward(self,x):
        out = F.relu(self.backbone(x), inplace=True)
        return out # 2208 channel


# if __name__ == "__main__":
    # model = densenet121(pretrained=True)
    # state_dict = model.state_dict()
    # torch.save(state_dict,"temp.pth")
    # inputs = torch.randn(1,3,256,256)
    # outputs = model(inputs)
    # print(outputs.shape)
if __name__ == "__main__":
    model = densenet169().state_dict()
    torch.save(model,"densenet169.pth")
    model = densenet201().state_dict()
    torch.save(model,"densenet201.pth")
    model = densenet161().state_dict()
    torch.save(model,"densenet161.pth")
