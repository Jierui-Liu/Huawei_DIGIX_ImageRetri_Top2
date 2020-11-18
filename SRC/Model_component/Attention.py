import torch
import torch.nn.functional as F
from torch import nn


class self_attention(nn.Module):
    def __init__(self,in_channels,kernel_size=(1,1)):
        super().__init__()


        self.attention_conv=nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=kernel_size)


    def forward(self,x):
        w=x.shape[2]
        h=x.shape[3]

        raw_attention=self.attention_conv(x)
        attention_map=F.softmax(raw_attention.reshape(-1,w*h),dim=1).reshape(-1,1,w,h)
        attention_feature=torch.mul(x,attention_map)
        return attention_feature,attention_map

class smart_self_attention(nn.Module):
    def __init__(self,in_channels,mid_conv_channel=512,input_size=[32,32]):
        super().__init__()
        self.w=input_size[0]
        self.h=input_size[1]

        self.attention_conv=nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=mid_conv_channel,kernel_size=(1,1)),
                                          nn.BatchNorm2d(mid_conv_channel),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(in_channels=mid_conv_channel,out_channels=1,kernel_size=(3,3),padding=1))


    def forward(self,x):
        raw_attention=self.attention_conv(x)
        attention_map=F.softmax(raw_attention.reshape(-1,self.w*self.h),dim=1).reshape(-1,1,self.w,self.h)
        attention_feature=torch.mul(x,attention_map)
        return attention_feature,attention_map
