# -*- coding: utf-8 -*-

import torch

from ..metric_base import MetricBase
from ...registry import METRICS

from typing import Dict

@METRICS.register
class KNN(MetricBase):
    """
    Similarity measure based on the euclidean distance.

    Hyper-Params:
        top_k (int): top_k nearest neighbors will be output in sorted order. If it is 0, all neighbors will be output.
    """
    default_hyper_params = {
        "top_k": 0,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(KNN, self).__init__(hps)

    def _cal_dis(self, query_fea: torch.tensor, gallery_fea: torch.tensor) -> torch.tensor:
        """
        Calculate the distance between query set features and gallery set features.

        Args:
            query_fea (torch.tensor): query set features.
            gallery_fea (torch.tensor): gallery set features.

        Returns:
            dis (torch.tensor): the distance between query set features and gallery set features.
        """
        query_fea = query_fea.transpose(1, 0)
        inner_dot = gallery_fea.mm(query_fea)
        dis = (gallery_fea ** 2).sum(dim=1, keepdim=True) + (query_fea ** 2).sum(dim=0, keepdim=True)
        dis = dis - 2 * inner_dot
        dis = dis.transpose(1, 0)

        # len_g=gallery_fea.shape[0]
        # len_q=query_fea.shape[0]
        # inputs=torch.cat((gallery_fea,query_fea),dim=0)
        # n = inputs.size(0)
        # dis = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)  # 每个数平方后， 进行加和（通过keepdim保持2维），再扩展成nxn维
        # dis = dis + dis.t()  # 这样每个dis[i][j]代表的是第i个特征与第j个特征的平方的和
        # dis.addmm_(1, -2, inputs, inputs.t())  # 然后减去2倍的 第i个特征*第j个特征 从而通过完全平方式得到 (a-b)^2
        # dis = dis.clamp(min=1e-12).sqrt()  # 然后开方
        # dis=dis[-len_q:,:len_g]

        print('query_fea.shape:',query_fea.shape)
        print('gallery_fea.shape:',gallery_fea.shape)
        print('dis.shape:',dis.shape)
        return dis

    def __call__(self, query_fea: torch.tensor, gallery_fea: torch.tensor) -> (torch.tensor, torch.tensor):

        dis = self._cal_dis(query_fea, gallery_fea)
        sorted_index = torch.argsort(dis, dim=1)
        if self._hyper_params["top_k"] != 0:
            sorted_index = sorted_index[:, :self._hyper_params["top_k"]]
        return dis, sorted_index
