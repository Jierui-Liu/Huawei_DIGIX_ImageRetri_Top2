'''
Author: your name
Date: 2020-08-11 10:40:53
LastEditTime: 2020-08-16 23:43:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PyRetri-master-gai1/pyretri/index/re_ranker/re_ranker_impl/kr_qe.py
'''
# -*- coding: utf-8 -*-

import torch

from ...metric import MetricBase
from .query_expansion import QE
from .fast_k_reciprocal import Fast_KReciprocal
from ..re_ranker_base import ReRankerBase
from ...registry import RERANKERS

from typing import Dict

@RERANKERS.register
class KRQE(ReRankerBase):
    """
    Apply query expansion and k-reciprocal.

    Hyper-Params:
        qe_times (int): number of query expansion times.
        qe_k (int): number of the neighbors to be combined.
        k1 (int): hyper-parameter for calculating jaccard distance.
        k2 (int): hyper-parameter for calculating local query expansion.
        lambda_value (float): hyper-parameter for calculating the final distance.
    """
    default_hyper_params = {
        "qe_times": 1,
        "qe_k": 1,
        "k1": 20,
        "k2": 6,
        "lambda_value": 0.3,
        "N":6000,
        "dist_type":"euclidean_distance"
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(KRQE, self).__init__(hps)
        qe_hyper_params = {
            "qe_times": self._hyper_params["qe_times"],
            "qe_k": self.default_hyper_params["qe_k"],
        }
        kr_hyper_params = {
            "k1" : self._hyper_params["k1"],
            "k2" : self._hyper_params["k2"],
            "lambda_value": self._hyper_params["lambda_value"],
            "dist_type": self._hyper_params["dist_type"],
            "N" : self._hyper_params["N"],
        }
        self.qe = QE(hps=qe_hyper_params)
        self.kr = Fast_KReciprocal(hps=kr_hyper_params)

    # def __call__(self, query_fea: torch.tensor, gallery_fea: torch.tensor, dis: torch.tensor or None = None,
    def __call__(self, query_fea: torch.tensor, gallery_fea: torch.tensor,metric:MetricBase, dis: torch.tensor or None = None,
                 sorted_index: torch.tensor or None = None) -> torch.tensor:

        # sorted_index = self.qe(query_fea, gallery_fea, dis, kr=self.kr)
        # jerry
        sorted_index = self.kr(query_fea, gallery_fea,metric, dis)
        sorted_index=torch.Tensor(sorted_index).long()
        sorted_index = self.qe(query_fea, gallery_fea,metric, sorted_index=sorted_index)
        return sorted_index

