'''
@Author      : now more
@Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
@Description : 
LastEditTime: 2020-08-24 23:57:01
'''

import torch
import torch.nn.functional as F
from torch import nn

from typing import Dict, List



class AdaptiveAvgPool2d(nn.Module):
    def __init__(self,output_size=(1,1),flatten=False):
        super(AdaptiveAvgPool2d, self).__init__()
        self.flatten = flatten
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=output_size)
    def forward(self, x):
        x = self.avgpool(x)
        # if self.flatten:
        #     x = torch.flatten(x, 1)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        
class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """
    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.avgpool = FastGlobalAvgPool2d()

    def forward(self, x):
        x_avg = self.avgpool(x, self.output_size)
        x_max = F.adaptive_max_pool2d(x, 1)
        x = x_max + x_avg
        return x


class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class MultiBranchPool(nn.Module):
    def __init__(self,reduce_dim=256):
        super(MultiBranchPool,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.fc_global = nn.Conv2d(2048,256,1)

        self.fc_hori_2_1 = nn.Conv2d(2048,256,1)
        self.fc_hori_2_2 = nn.Conv2d(2048,256,1)
        self.fc_hori_3_1 = nn.Conv2d(2048,256,1)
        self.fc_hori_3_2 = nn.Conv2d(2048,256,1)
        self.fc_hori_3_3 = nn.Conv2d(2048,256,1)

        self.fc_vert_2_1 = nn.Conv2d(2048,256,1)
        self.fc_vert_2_2 = nn.Conv2d(2048,256,1)
        self.fc_vert_3_1 = nn.Conv2d(2048,256,1)
        self.fc_vert_3_2 = nn.Conv2d(2048,256,1)
        self.fc_vert_3_3 = nn.Conv2d(2048,256,1)

        self.fc_annu_2_1 = nn.Conv2d(2048,256,1)
        self.fc_annu_2_2 = nn.Conv2d(2048,256,1)
        self.fc_annu_3_1 = nn.Conv2d(2048,256,1)
        self.fc_annu_3_2 = nn.Conv2d(2048,256,1)
        self.fc_annu_3_3 = nn.Conv2d(2048,256,1)



    def forward(self,x): # x.shape batch*channels*12*12
        global_branch = torch.flatten(self.fc_global(self.avgpool(x)),1)

        hori_2_1 = torch.flatten(self.fc_hori_2_1(self.avgpool(x[:,:,:,:6])),1)
        hori_2_2 = torch.flatten(self.fc_hori_2_2(self.avgpool(x[:,:,:,6:])),1)
        hori_2 = torch.cat((hori_2_1,hori_2_2),1)
        hori_3_1 = torch.flatten(self.fc_hori_3_1(self.avgpool(x[:,:,:,:4])),1)
        hori_3_2 = torch.flatten(self.fc_hori_3_2(self.avgpool(x[:,:,:,4:8])),1)
        hori_3_3 = torch.flatten(self.fc_hori_3_3(self.avgpool(x[:,:,:,8:])),1)
        hori_3 = torch.cat((hori_3_1,hori_3_2,hori_3_3),dim=1)

        vert_2_1 = torch.flatten(self.fc_vert_2_1(self.avgpool(x[:,:,:6,:])),1)
        vert_2_2 = torch.flatten(self.fc_vert_2_2(self.avgpool(x[:,:,6:,:])),1)
        vert_2 = torch.cat((vert_2_1,vert_2_2),dim=1)
        vert_3_1 = torch.flatten(self.fc_vert_3_1(self.avgpool(x[:,:,:4,:])),1)
        vert_3_2 = torch.flatten(self.fc_vert_3_2(self.avgpool(x[:,:,4:8,:])),1)
        vert_3_3 = torch.flatten(self.fc_vert_3_3(self.avgpool(x[:,:,8:,:])),1)
        vert_3 = torch.cat((vert_3_1,vert_3_2,vert_3_3),dim=1)

        annu_2_1 = torch.flatten(self.fc_annu_2_1(self.avgpool(x[:,:,3:9,3:9])),1)
        temp = x.clone()
        temp[:,:,3:9,3:9] = 0
        annu_2_2 = torch.flatten(self.fc_annu_2_2(self.avgpool(temp)*4/3),1)
        annu_2 = torch.cat((annu_2_1,annu_2_2),dim=1)

        annu_3_1 = torch.flatten(self.fc_annu_3_1(self.avgpool(x[:,:,4:8,4:8])),1)
        temp = x[:,:,2:10,2:10].clone()
        temp[:,:,2:6,2:6] = 0
        annu_3_2 = torch.flatten(self.fc_annu_3_2(self.avgpool(temp)*4/3),1)
        temp = x.clone()
        temp[:,:,2:10,2:10] = 0
        annu_3_3 = torch.flatten(self.fc_annu_3_3(self.avgpool(temp)*4/3),1)
        annu_3 = torch.cat((annu_3_1,annu_3_2,annu_3_3),dim=1)

        return global_branch,hori_2,hori_3,vert_2,vert_3,annu_2,annu_3




class SPoC(nn.Module):
    """
    SPoC with center prior.
    c.f. https://arxiv.org/pdf/1510.07493.pdf
    """

    def __init__(self):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(SPoC, self).__init__()
        self.first_show = True
        self.spatial_weight_cache = dict()

    def forward(self, fea: torch.tensor) -> torch.tensor:

        h, w = fea.shape[2:]
        sigma = min(h, w) / 2.0 / 3.0
        x = torch.Tensor(range(w))
        y = torch.Tensor(range(h))[:, None]
        spatial_weight = torch.exp(-((x - (w - 1) / 2.0) ** 2 + (y - (h - 1) / 2.0) ** 2) / 2.0 / (sigma ** 2))
        spatial_weight = spatial_weight.cuda()
        spatial_weight = spatial_weight[None, None, :, :]
        fea = (fea * spatial_weight).sum(dim=(2, 3),keepdims=True)
        return fea


class Crow(nn.Module):
    """
    Cross-dimensional Weighting for Aggregated Deep Convolutional Features.
    c.f. https://arxiv.org/pdf/1512.04065.pdf

    Hyper-Params
        spatial_a (float): hyper-parameter for calculating spatial weight.
        spatial_b (float): hyper-parameter for calculating spatial weight.
    """

    def __init__(self, spatial_a=2.0,spatial_b=2.0):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(Crow, self).__init__()
        self.spatial_a=spatial_a
        self.spatial_b=spatial_b

    def forward(self, fea: torch.tensor) -> torch.tensor:

        spatial_weight = fea.detach().sum(dim=1, keepdims=True)
        z = (spatial_weight ** self.spatial_a).sum(dim=(2, 3), keepdims=True)
        z = z ** (1.0 / self.spatial_a)
        spatial_weight = (spatial_weight / z) ** (1.0 / self.spatial_b)

        c, w, h = fea.shape[1:]
        nonzeros = (fea!=0).float().sum(dim=(2, 3), keepdims=True) / 1.0 / (w * h) + 1e-6
        channel_weight = torch.log(nonzeros.sum(dim=1, keepdims=True) / nonzeros)

        fea = fea * spatial_weight
        fea = fea.sum(dim=(2, 3), keepdims=True)
        fea = fea * channel_weight

        return fea


class GMP(nn.Module):
    """
    Global maximum pooling
    """

    def __init__(self):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(GMP, self).__init__()

    def forward(self, fea: torch.tensor) -> torch.tensor:
        
        fea = (fea.max(dim=3,keepdims=True)[0]).max(dim=2,keepdims=True)[0]
        
        return fea


class RMAC(nn.Module):

    def __init__(self):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(RMAC, self).__init__()
        self.first_show = True
        self.cached_regions = dict()
        self.level_n=3


    def _get_regions(self, h: int, w: int) -> List:
        """
        Divide the image into several regions.

        Args:
            h (int): height for dividing regions.
            w (int): width for dividing regions.

        Returns:
            regions (List): a list of region positions.
        """
        if (h, w) in self.cached_regions:
            return self.cached_regions[(h, w)]

        m = 1
        n_h, n_w = 1, 1
        regions = list()
        if h != w:
            min_edge = min(h, w)
            left_space = max(h, w) - min(h, w)
            iou_target = 0.4
            iou_best = 1.0
            while True:
                iou_tmp = (min_edge ** 2 - min_edge * (left_space // m)) / (min_edge ** 2)

                # small m maybe result in non-overlap
                if iou_tmp <= 0:
                    m += 1
                    continue

                if abs(iou_tmp - iou_target) <= iou_best:
                    iou_best = abs(iou_tmp - iou_target)
                    m += 1
                else:
                    break
            if h < w:
                n_w = m
            else:
                n_h = m

        for i in range(self.level_n):
            region_width = int(2 * 1.0 / (i + 2) * min(h, w))
            step_size_h = (h - region_width) // n_h
            step_size_w = (w - region_width) // n_w

            for x in range(n_h):
                for y in range(n_w):
                    st_x = step_size_h * x
                    ed_x = st_x + region_width - 1
                    assert ed_x < h
                    st_y = step_size_w * y
                    ed_y = st_y + region_width - 1
                    assert ed_y < w
                    regions.append((st_x, st_y, ed_x, ed_y))

            n_h += 1
            n_w += 1

        self.cached_regions[(h, w)] = regions
        return regions

    def forward(self, fea: torch.tensor) -> torch.tensor:
        
        h, w = fea.shape[2:]
        final_fea = None
        regions = self._get_regions(h, w)
        for _, r in enumerate(regions):
            st_x, st_y, ed_x, ed_y = r
            region_fea = (fea[:, :, st_x: ed_x, st_y: ed_y].max(dim=3)[0]).max(dim=2)[0]
            region_fea = region_fea / torch.norm(region_fea, dim=1, keepdim=True)
            if final_fea is None:
                final_fea = region_fea
            else:
                final_fea = final_fea + region_fea
            
        return final_fea.unsqueeze(2).unsqueeze(3)






if __name__ == "__main__":
    inputs = torch.randn(2,2048,12,12)
    model = SPoC()
    outputs = model(inputs)
    print(output.shape)