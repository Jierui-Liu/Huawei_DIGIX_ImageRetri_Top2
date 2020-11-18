# -*- coding: utf-8 -*-

import torch
import numpy as np

from ...metric import MetricBase
from ..re_ranker_base import ReRankerBase
from ...registry import RERANKERS

import time
from typing import Dict

import gc
from tqdm import tqdm

import pynvml 
import GPUtil
import os
from random import shuffle

def get_gpu(num_of_gpu):
    # 获取GPU，并且设置环境变量，并且返回可以使用的GPU编号
    # 注意，最好在运行这个函数之后立刻占用获取的GPU，否则GPU可能被其他程序占用，导致获取失败。

    gpu_ids_avail = GPUtil.getAvailable(maxMemory=0.02, limit=8)

    if len(gpu_ids_avail) < num_of_gpu:
        #如果正确的提交了任务，不应该获取不到足够的GPU
        #queue.pl -q GPU_QUEUE --num-threads 4  #需要占用4个GPU
        print("not enough GPU")
        return []

    shuffle(gpu_ids_avail)
    CUDA_VISIBLE_DEVICES=""
    for gpu in gpu_ids_avail[:num_of_gpu]:
        CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES+str(gpu)+","

    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES[:-1]
    print("using GPU:",CUDA_VISIBLE_DEVICES[:-1])

    return list(range(num_of_gpu))


@RERANKERS.register
class Fast_KReciprocal_Finaldist(ReRankerBase):

    default_hyper_params = {
        "k1": 20,
        "k2": 6,
        "lambda_value": 0.3,
        "N":6000,
        "dist_num_out": 20,
        "dist_type":"euclidean_distance"
    }

    def __init__(self, hps: Dict or None = None):
        super(Fast_KReciprocal_Finaldist, self).__init__(hps)

    def euclidean_distance(self,qf, gf):
        # m = qf.shape[0]
        # n = gf.shape[0]

        # # for L2-norm feature
        # dist_mat = 2 - 2 * torch.matmul(qf, gf.t())
        qf = qf.transpose(1, 0)
        inner_dot = gf.mm(qf)
        dis = (gf ** 2).sum(dim=1, keepdim=True) + (qf ** 2).sum(dim=0, keepdim=True)
        dis = dis - 2 * inner_dot
        dis = dis.transpose(1, 0)
        return dis

    
    def similarity_distance(self, qf, gf):
        # m = qf.shape[0]
        # n = gf.shape[0]

        dis = 1-torch.matmul(qf, torch.t(gf))
        return dis

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

    # def __call__(self,probFea, galFea,dis: torch.tensor or None = None,
    #              sorted_index: torch.tensor or None = None):
    def __call__(self, query_fea: torch.tensor, gallery_fea: torch.tensor,metric:MetricBase,  dis: torch.tensor or None = None,
                 sorted_index: torch.tensor or None = None) -> torch.tensor or np.ndarray:
        # The following naming, e.g. gallery_num, is different from outer scope.
        # Don't care about it.
        self.k1 = self._hyper_params["k1"]
        self.k2 = self._hyper_params["k2"]
        self.lambda_value = self._hyper_params["lambda_value"]
        self.dist_type = self._hyper_params["dist_type"]
        self.N = self._hyper_params["N"]
        
        k1 = self.k1
        k2 = self.k2
        lambda_value = self.lambda_value
        N = self.N
        
        t1 = time.time()
        query_num = query_fea.size(0)
        all_num = query_num + gallery_fea.size(0)
        feat = torch.cat([query_fea, gallery_fea])
        
        device = get_gpu(1)
        feat = feat.cuda()

        initial_rank = self.batch_torch_topk(feat, feat)
        # del feat
        del query_fea
        del gallery_fea
        torch.cuda.empty_cache()  # empty GPU memory
        gc.collect()  # empty memory
        print('Using totally {:.2f}s to compute initial_rank'.format(time.time() - t1))
        print('starting re_ranking')

        R = []
        for i in tqdm(range(all_num)):
            # k-reciprocal neighbors
            k_reciprocal_index = self.k_reciprocal_neigh(initial_rank, i, k1)
            k_reciprocal_expansion_index = k_reciprocal_index
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_k_reciprocal_index = self.k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            R.append(k_reciprocal_expansion_index)

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
        '''gc.collect()  # empty memory
        N = 2000
        for j in range(all_num // N + 1):

            if k2 != 1:
                V_qe = np.zeros_like(V[:, j * N:j * N + N], dtype=np.float32)
                for i in range(all_num):
                    V_qe[i, :] = np.mean(V[initial_rank[i], j * N:j * N + N], axis=0)
                V[:, j * N:j * N + N] = V_qe
                del V_qe
        del initial_rank'''

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

        dist_num_out=self._hyper_params["dist_num_out"]
        final_dist = final_dist[:query_num, query_num:]
        sorted_idx = np.argsort(final_dist,axis=1)[:,:dist_num_out]
        final_dist=[final_dist[index_dist][sorted_idx[index_dist,:dist_num_out]] for index_dist in range(len(final_dist))]
        print('Using totally {:.2f}S to compute final_distance'.format(time.time() - t1))
        # return final_dist
        return sorted_idx,final_dist



if __name__ == "__main__":
    # kr = KReciprocal()
    my_kr = Fast_KReciprocal()