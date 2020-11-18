import torch
import torch.nn as nn
import math

class Embedding_dropout(nn.Module):
    def __init__(self,p=0.4):  #P的概率变成0
        super().__init__()
        self.p=p
        self.correction=math.sqrt(1-self.p)   #乘以dropout后的embedding

    def forward(self,embeddings):        #(batchsize,embedding_dim)
        embedding_dim=embeddings.shape[1]
        minibatchsize=embeddings.shape[0]
        mask=(torch.rand((1,embedding_dim))>self.p)   #p是0的概率，所以mask中False的地方因为0
        mask=mask.expand( minibatchsize,embedding_dim)

        embeddings[mask==False]=0

        return embeddings*self.correction


if __name__ == '__main__':
    seed=20
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    emd=Embedding_dropout(0.9)
    test_x=torch.randn((3,1664))
    test=emd(test_x)

    a=1