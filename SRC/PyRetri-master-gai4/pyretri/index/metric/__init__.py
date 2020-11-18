# -*- coding: utf-8 -*-

from yacs.config import CfgNode

from .metric_impl.knn_eula import KNN_EULA
from .metric_impl.knn import KNN
# from .metric_impl.PLDA import PLDA
from .metric_base import MetricBase


__all__ = [
    'MetricBase',
    'KNN',
    'PLDA',
    'KNN_EULA',
]
