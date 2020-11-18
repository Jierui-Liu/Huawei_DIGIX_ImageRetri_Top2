# -*- coding: utf-8 -*-

import numpy as np

from ..dim_processors_base import DimProcessorBase
from ...registry import DIMPROCESSORS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle

from typing import Dict, List


@DIMPROCESSORS.register
class LDA_my(DimProcessorBase):
    """
    Do the SVD transformation for dimension reduction.

    Hyper-Params:
        proj_dim (int):  the dimension after reduction. If it is 0, then no reduction will be done
            (in SVD, we will minus origin dimension by 1).
        whiten (bool): whether do whiten for each part.
        train_fea_dir (str): the path of features for training SVD.
        l2 (bool): whether do l2-normalization for the training features.
    """
    default_hyper_params = {
        "proj_dim": 0,
        "train_fea_dir": "unknown",
    }

    def __init__(self, feature_names: List[str], hps: Dict or None = None):
        """
        Args:
            feature_names (list): a list of features names to be loaded.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(LDA_my, self).__init__(feature_names, hps)
        
        f = open(self._hyper_params["train_fea_dir"], "rb")
        dict_data = pickle.load(f)
        f.close()
        labels=dict_data["fname"]
        train=dict_data["data"]

        self.LDA_model=LDA(n_components=self._hyper_params["proj_dim"])
        self.LDA_model.fit(train,labels)


    def __call__(self, fea: np.ndarray) -> np.ndarray:
        ori_fea = fea
        proj_fea = self.LDA_model.transform(ori_fea)
        return proj_fea

