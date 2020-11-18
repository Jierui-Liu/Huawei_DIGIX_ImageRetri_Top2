'''
Author: your name
Date: 2020-08-13 23:13:13
LastEditTime: 2020-08-14 02:45:48
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /HW2/SRC/Model_component/XBM.py
'''
import torch

def pair_based_loss(dist,y,all_label,ranking_loss):
    dist_ap, dist_an = [], []
    n = y.size(0) 
    n_all = all_label.size(0) 
    y=y.view(n,1)
    all_label=all_label.view(1,n_all)
    mask=y.expand(n,n_all).eq(all_label.expand(n,n_all))
    for i in range(n):
        if len(dist[i][mask[i]])>0 and len(dist[i][mask[i] == 0])>0:
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # 在i与所有有相同label的j的距离中找一个最大的
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # 在i与所有不同label的j的距离找一个最小的
    dist_ap = torch.cat(dist_ap)  # 将list里的tensor拼接成新的tensor
    dist_an = torch.cat(dist_an)
 
    # Compute ranking hinge loss
    y = torch.ones_like(dist_an)  # 声明一个与dist_an相同shape的全1tensor

    loss = ranking_loss(dist_an, dist_ap, y)
    return loss


class XBM:
    def __init__(self, num_batch=100):
        # init memory
        self.feats = list()
        self.labels = list()
        self.indices = list()
        self.num_batch=num_batch
        self.num_batch_saved=0
        

    def enqueue_dequeue(self, feats, labels):
        if self.num_batch_saved==0:
            self.feats=feats
            self.labels=labels
            self.num_batch_saved+=1
        elif self.num_batch_saved<self.num_batch:
            self.feats=torch.cat((self.feats.detach(),feats),dim=0)
            self.labels=torch.cat((self.labels.detach(),labels))
            self.num_batch_saved+=1
        else:
            n = feats.size(0) 
            self.feats=torch.cat((self.feats.detach(),feats),dim=0)
            self.labels=torch.cat((self.labels.detach(),labels))
            self.feats=self.feats[n:]
            self.labels=self.labels[n:]

            
    def get(self):
        return self.feats, self.labels