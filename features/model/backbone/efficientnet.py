'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-10-31 09:20:38
'''
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class efficient_b4(nn.Module):
    def __init__(self):
        super(efficient_b4,self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
    def forward(self,inputs):
        return self.model.extract_features(inputs)

class efficient_b5(nn.Module):
    def __init__(self):
        super(efficient_b5,self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=2)
    def forward(self,inputs):
        return self.model.extract_features(inputs)

class efficient_b3(nn.Module):
    def __init__(self):
        super(efficient_b3,self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)
    def forward(self,inputs):
        return self.model.extract_features(inputs)
        


if __name__ == "__main__":
    inputs = torch.randn(1,3,224,224)
    model = efficient_b5()
    outputs = model(inputs)
    print(outputs.shape)
    length_backbone = len(list(model.named_children()))
    print("len_layers : ",length_backbone)
    state_dict = model.state_dict()
    for key in state_dict.keys():
        print(key)