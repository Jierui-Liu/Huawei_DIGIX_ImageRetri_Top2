import torch
import torch.nn as nn

import torch.nn.functional as F

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

class horizen_vertical_pool(nn.Module):
    #来自JD比赛TOP1的网络结构，但是只有垂直池化，水平池化和全局池化。
    def __init__(self, reduce_dim=256):
        super(MultiBranchPool, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc_global = nn.Conv2d(2048, 256, 1)

        self.fc_hori_2_1 = nn.Conv2d(2048, 256, 1)
        self.fc_hori_2_2 = nn.Conv2d(2048, 256, 1)
        self.fc_hori_3_1 = nn.Conv2d(2048, 256, 1)
        self.fc_hori_3_2 = nn.Conv2d(2048, 256, 1)
        self.fc_hori_3_3 = nn.Conv2d(2048, 256, 1)

        self.fc_vert_2_1 = nn.Conv2d(2048, 256, 1)
        self.fc_vert_2_2 = nn.Conv2d(2048, 256, 1)
        self.fc_vert_3_1 = nn.Conv2d(2048, 256, 1)
        self.fc_vert_3_2 = nn.Conv2d(2048, 256, 1)
        self.fc_vert_3_3 = nn.Conv2d(2048, 256, 1)


    def forward(self, x):  # x.shape batch*channels*12*12
        global_branch = torch.flatten(self.fc_global(self.avgpool(x)), 1)

        hori_2_1 = torch.flatten(self.fc_hori_2_1(self.avgpool(x[:, :, :, :6])), 1)
        hori_2_2 = torch.flatten(self.fc_hori_2_2(self.avgpool(x[:, :, :, 6:])), 1)
        hori_2 = torch.cat((hori_2_1, hori_2_2), 1)
        hori_3_1 = torch.flatten(self.fc_hori_3_1(self.avgpool(x[:, :, :, :4])), 1)
        hori_3_2 = torch.flatten(self.fc_hori_3_2(self.avgpool(x[:, :, :, 4:8])), 1)
        hori_3_3 = torch.flatten(self.fc_hori_3_3(self.avgpool(x[:, :, :, 8:])), 1)
        hori_3 = torch.cat((hori_3_1, hori_3_2, hori_3_3), dim=1)

        vert_2_1 = torch.flatten(self.fc_vert_2_1(self.avgpool(x[:, :, :6, :])), 1)
        vert_2_2 = torch.flatten(self.fc_vert_2_2(self.avgpool(x[:, :, 6:, :])), 1)
        vert_2 = torch.cat((vert_2_1, vert_2_2), dim=1)
        vert_3_1 = torch.flatten(self.fc_vert_3_1(self.avgpool(x[:, :, :4, :])), 1)
        vert_3_2 = torch.flatten(self.fc_vert_3_2(self.avgpool(x[:, :, 4:8, :])), 1)
        vert_3_3 = torch.flatten(self.fc_vert_3_3(self.avgpool(x[:, :, 8:, :])), 1)
        vert_3 = torch.cat((vert_3_1, vert_3_2, vert_3_3), dim=1)

        return global_branch, hori_2, hori_3, vert_2, vert_3




class statistics_pool(nn.Module):
    def __init__(self,eps=1e-5):
        super().__init__()
        self.eps=eps
    def forward(self,x):
        #x[B,C,W,H]
        x=torch.flatten(x, start_dim=2, end_dim=3)
        x_mean=x.mean(dim=-1,keepdim=True)
        x_var = torch.sqrt((x - x_mean).pow(2).mean(-1) + self.eps)
        return torch.cat([x_mean.squeeze(-1), x_var], -1)



class voting_pool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x_max=x.max(-1, dim=(2,3),keepdim=True)
        b=1












        a,indices=F.adaptive_max_pool2d_with_indices(x,output_size=(1,1),return_indices=True)
        c=x.shape[1]
        w=x.shape[2]
        h=x.shape[3]
        x=torch.reshape(x,[-1,c,w*h])


        indices_unique = indices.unique(sorted=True,dim=0,return_counts=True)
        indices_unique_count = torch.stack([(indices == x_u).sum() for x_u in indices_unique])


        x=torch.index_select(x,1,indices[:])


        return x
if __name__ == '__main__':
    import numpy as np

    b=np.ones((2,3,5,5))
    b[0,0,0,0]=2

    b[0,2,1,0]=3

    b[0,1,1,1]=5
    b[0,0,1,1]=7

    b[0,2,2,4]=69
    a=torch.tensor(b)

    sp=statistics_pool()
    out=sp(a)



    vp=voting_pool()
    out=vp(a)

