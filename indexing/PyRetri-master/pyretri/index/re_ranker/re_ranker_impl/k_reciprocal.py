# -*- coding: utf-8 -*-

import torch
import numpy as np

from ..re_ranker_base import ReRankerBase
from ...registry import RERANKERS

import time
from typing import Dict

import gc
from tqdm import tqdm

import pynvml 

def get_free_device_ids():
    pynvml.nvmlInit()
    num_device = pynvml.nvmlDeviceGetCount()
    free_device_id = []
    for i in range(num_device):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        men_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(men_info.total,men_info.free)
        # import pdb; pdb.set_trace()
        if men_info.free >= men_info.total*0.3:
            free_device_id.append(i)
    return free_device_id

@RERANKERS.register
class Fast_KReciprocal():

    default_hyper_params = {
        "k1": 20,
        "k2": 6,
        "lambda_value": 0.3,
    }

    def __init__(self,hps=None,k1=25,k2=6,lambda_value=0.5,N=2000,dist_type="euclidean_distance"):
        # super(Fast_KReciprocal, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.lambda_value = lambda_value
        self.dist_type = dist_type
        self.N = N

    def euclidean_distance(self,qf,gf):
        qf = qf.transpose(1,0)
        inner_dot = gf.mm(qf)
        dis = (gf**2).sum(dim=1,keepdim=True) + (qf**2).sum(dim=0,keepdim=True)
        dis = dis - 2*inner_dot
        dis = dis.transpose(1, 0)
        return dis

    # def euclidean_distance(self,qf, gf):
    #     m = qf.shape[0]
    #     n = gf.shape[0]

    #     # for L2-norm feature
    #     dist_mat = 2 - 2 * torch.matmul(qf, gf.t())
    #     return dist_mat

    def batch_euclidean_distance(self,qf, gf):
        N = self.N
        dist_func = getattr(self,self.dist_type)
        m = qf.shape[0]
        n = gf.shape[0]

        dist_mat = []
        for j in range(n // N + 1):
            temp_gf = gf[j * N:j * N + N]
            temp_qd = []
            for i in range(m // N + 1):
                temp_qf = qf[i * N:i * N + N]
                temp_d = dist_func(temp_qf, temp_gf)
                temp_qd.append(temp_d)
            temp_qd = torch.cat(temp_qd, dim=0)
            temp_qd = temp_qd / (torch.max(temp_qd, dim=0)[0])
            dist_mat.append(temp_qd.t().cpu())
        del temp_qd
        del temp_gf
        del temp_qf
        del temp_d
        torch.cuda.empty_cache()  # empty GPU memory
        dist_mat = torch.cat(dist_mat, dim=0)
        return dist_mat


    def batch_torch_topk(self,qf, gf):
        # 将topK排序放到GPU里运算，并且只返回k1+1个结果
        # Compute TopK in GPU and return (k1+1) results
        m = qf.shape[0]
        n = gf.shape[0]
        k1 = self.k1
        N = self.N
        dist_func = getattr(self,self.dist_type)
        torch.cuda.empty_cache()  # empty GPU memory

        dist_mat = []
        initial_rank = []
        for j in range(n // N + 1):
            temp_gf = gf[j * N:j * N + N]
            temp_qd = []
            for i in range(m // N + 1):
                temp_qf = qf[i * N:i * N + N]
                temp_d = dist_func(temp_qf, temp_gf)
                temp_qd.append(temp_d)
            temp_qd = torch.cat(temp_qd, dim=0)
            temp_qd = temp_qd / (torch.max(temp_qd, dim=0)[0])
            temp_qd = temp_qd.t()
            initial_rank.append(torch.topk(temp_qd, k=k1, dim=1, largest=False, sorted=True)[1])

        del temp_qd
        del temp_gf
        del temp_qf
        del temp_d
        torch.cuda.empty_cache()  # empty GPU memory
        initial_rank = torch.cat(initial_rank, dim=0).cpu().numpy()
        return initial_rank

    def batch_v(self,feat, R, all_num):
        dist_func = getattr(self,self.dist_type)
        V = np.zeros((all_num, all_num), dtype=np.float32)
        m = feat.shape[0]
        for i in tqdm(range(m)):
            temp_gf = feat[i].unsqueeze(0)
            # temp_qd = []
            temp_qd = dist_func(temp_gf, feat)
            temp_qd = temp_qd / (torch.max(temp_qd))
            temp_qd = temp_qd.squeeze()
            temp_qd = temp_qd[R[i]]
            weight = torch.exp(-temp_qd)
            weight = (weight / torch.sum(weight)).cpu().numpy()
            V[i, R[i]] = weight.astype(np.float32)
        return V

    def k_reciprocal_neigh(self,initial_rank, i, k1):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        return forward_k_neigh_index[fi]

    def __call__(self,probFea, galFea,dis: torch.tensor or None = None,
                 sorted_index: torch.tensor or None = None):
        # The following naming, e.g. gallery_num, is different from outer scope.
        # Don't care about it.
        k1 = self.k1
        k2 = self.k2
        lambda_value = self.lambda_value
        print("lambda value",lambda_value)
        print("k1 : {}, k2 : {}".format(k1,k2))
        print("N : {}".format(self.N))
        N = self.N
        
        t1 = time.time()
        query_num = probFea.size(0)
        all_num = query_num + galFea.size(0)
        feat = torch.cat([probFea, galFea])
        
        if len(get_free_device_ids()):
            device = get_free_device_ids()[0]
            # print(device)
            # feat = feat.cuda(device)
            feat = feat.cuda()


        initial_rank = self.batch_torch_topk(feat, feat)
        # del feat
        del probFea
        del galFea
        torch.cuda.empty_cache()  # empty GPU memory
        gc.collect()  # empty memory
        print('Using totally {:.2f}s to compute initial_rank'.format(time.time() - t1))
        print('starting re_ranking')

        R = []
        # cnt_list = [] #
        for i in tqdm(range(all_num)):
            # k-reciprocal neighbors
            k_reciprocal_index = self.k_reciprocal_neigh(initial_rank, i, k1)
            k_reciprocal_expansion_index = k_reciprocal_index
            # cnt = 0 #
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_k_reciprocal_index = self.k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
                # if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                #         candidate_k_reciprocal_index):
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 5 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
                    # cnt += 1 #
            # cnt_list.append(cnt) #
            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            R.append(k_reciprocal_expansion_index)
        # print("cnt finish")
        # import pandas as pd
        # df = pd.DataFrame(cnt_list)
        # df.to_csv("cnt_list.csv",index=None,header=None)
        gc.collect()  # empty memory
        print('Using totally {:.2f}S to compute R'.format(time.time() - t1))
        V = self.batch_v(feat, R, all_num)
        del R
        gc.collect()  # empty memory
        print('Using totally {:.2f}S to compute V-1'.format(time.time() - t1))
        initial_rank = initial_rank[:, :k2]

        ### 下面这个版本速度更快
        ### Faster version
        if k2 != 1:
            V_qe = np.zeros_like(V, dtype=np.float16)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i], :], axis=0)
            V = V_qe
            del V_qe
        del initial_rank

        ### 下面这个版本更省内存(约40%)，但是更慢
        ### Low-memory version
        # gc.collect()  # empty memory
        # N = 1000
        # for j in range(all_num // N + 1):

        #     if k2 != 1:
        #         V_qe = np.zeros_like(V[:, j * N:j * N + N], dtype=np.float32)
        #         for i in range(all_num):
        #             V_qe[i, :] = np.mean(V[initial_rank[i], j * N:j * N + N], axis=0)
        #         V[:, j * N:j * N + N] = V_qe
        #         del V_qe
        # del initial_rank

        gc.collect()  # empty memory
        print('Using totally {:.2f}S to compute V-2'.format(time.time() - t1))
        invIndex = []

        for i in range(all_num):
            invIndex.append(np.where(V[:, i] != 0)[0])
        print('Using totally {:.2f}S to compute invIndex'.format(time.time() - t1))

        jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)
        for i in tqdm(range(query_num)):
            temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
            indNonZero = np.where(V[i, :] != 0)[0]
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                                V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2. - temp_min)
        del V
        gc.collect()  # empty memory
        original_dist = self.batch_euclidean_distance(feat, feat[:query_num, :]).numpy()
        final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
        # print(jaccard_dist)
        del original_dist

        del jaccard_dist

        final_dist = final_dist[:query_num, query_num:]
        sorted_idx = np.argsort(final_dist,axis=1)[:,:50]
        print('Using totally {:.2f}S to compute final_distance'.format(time.time() - t1))
        # return final_dist
        return sorted_idx




@RERANKERS.register
class KReciprocal(ReRankerBase):
    """
    Encoding k-reciprocal nearest neighbors to enhance the performance of retrieval.
    c.f. https://arxiv.org/pdf/1701.08398.pdf

    Hyper-Params:
        k1 (int): hyper-parameter for calculating jaccard distance.
        k2 (int): hyper-parameter for calculating local query expansion.
        lambda_value (float): hyper-parameter for calculating the final distance.
    """
    default_hyper_params = {
        "k1": 20,
        "k2": 6,
        "lambda_value": 0.3,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(KReciprocal, self).__init__(hps)

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
        return dis

    def __call__(self, query_fea: torch.tensor, gallery_fea: torch.tensor,  dis: torch.tensor or None = None,
                 sorted_index: torch.tensor or None = None) -> torch.tensor or np.ndarray:
        # The following naming, e.g. gallery_num, is different from outer scope.
        # Don't care about it.
        q_g_dist = dis.cpu().numpy()
        g_g_dist = self._cal_dis(gallery_fea, gallery_fea).cpu().numpy()
        q_q_dist = self._cal_dis(query_fea, query_fea).cpu().numpy()

        original_dist = np.concatenate(
            [np.concatenate([q_q_dist, q_g_dist], axis=1),
             np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
            axis=0)
        original_dist = np.power(original_dist, 2).astype(np.float32)
        original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))
        V = np.zeros_like(original_dist).astype(np.float32)
        initial_rank = np.argsort(original_dist).astype(np.int32)

        query_num = q_g_dist.shape[0]
        gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
        all_num = gallery_num

        for i in range(all_num):
            # k-reciprocal neighbors
            forward_k_neigh_index = initial_rank[i, :self._hyper_params["k1"] + 1]
            backward_k_neigh_index = initial_rank[forward_k_neigh_index, :self._hyper_params["k1"] + 1]
            fi = np.where(backward_k_neigh_index == i)[0]
            k_reciprocal_index = forward_k_neigh_index[fi]
            k_reciprocal_expansion_index = k_reciprocal_index
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(self._hyper_params["k1"] / 2.)) + 1]
                candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                   :int(np.around(self._hyper_params["k1"] / 2.)) + 1]
                fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
                candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
            V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
        original_dist = original_dist[:query_num, ]
        if self._hyper_params["k2"] != 1:
            V_qe = np.zeros_like(V, dtype=np.float32)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i, :self._hyper_params["k2"]], :], axis=0)
            V = V_qe
            del V_qe
        del initial_rank
        invIndex = []
        for i in range(gallery_num):
            invIndex.append(np.where(V[:, i] != 0)[0])

        jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

        for i in range(query_num):
            temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
            indNonZero = np.where(V[i, :] != 0)[0]
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                                   V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

        final_dist = jaccard_dist * (1 - self._hyper_params["lambda_value"]) + original_dist * self._hyper_params[
            "lambda_value"]
        del original_dist, V, jaccard_dist
        final_dist = final_dist[:query_num, query_num:]

        
        # if torch.cuda.is_available():
        #     final_dist = torch.Tensor(final_dist).cuda()
        #     sorted_idx = torch.argsort(final_dist, dim=1)
        # else:
        #     sorted_idx = np.argsort(final_dist, axis=1)
        #jerry
        sorted_idx = np.argsort(final_dist, axis=1)
        return sorted_idx


if __name__ == "__main__":
    # kr = KReciprocal()
    my_kr = Fast_KReciprocal()