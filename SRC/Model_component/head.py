import torch
import torch.nn as nn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

class BNneckHead(nn.Module):
    def __init__(self,in_feat,num_classes):
        super(BNneckHead,self).__init__()
        self.bnneck = nn.BatchNorm2d(in_feat)
        self.bnneck.apply(weights_init_kaiming)
        self.bnneck.bias.requires_grad_(False)  # no shift
        
    
    def forward(self,features):
        return self.bnneck(features)[..., 0, 0]