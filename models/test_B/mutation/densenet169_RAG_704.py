
import torch
import torch.nn as nn
import torchvision
import sys
sys.path.append(".")
import SRC.pre_process.opencv_transoform as cvtransform
from SRC.During_training.utils import froze_all_param

from SRC.Model_component.CE_loss import AMLinear_dropout
import SRC.pre_process.denoise_transoform as denoise_transform
from SRC.Model_component.head import *
from SRC.Model_component.aggregator import *
from SRC.Model_component.XBM import *
from SRC.Model_component.RAG_module_init import RGA_Module
from SRC.Model_component.IBN import *



ce_tri_ratio=2
numofGPU=1
data_dir="/data_sda/LiuJierui/Dataset/HW_Retrieval/train_data"

#=====stage 1 config
stage1_images_per_classes=7
stage1_classes_per_minibatch=7
# stage1_numofstep=0
stage1_numofstep=50000
stage1_nums_addnegative=0
stage1_size=704
stage1_pre_process_train= torchvision.transforms.Compose([
        cvtransform.RescalePad(output_size=stage1_size),
        cvtransform.ShiftScaleRotate(p=0.3,shift_limit=0.1,scale_limit=(-0.5,0.2),rotate_limit=15),
        cvtransform.IAAPerspective(p=0.1,scale=(0.05, 0.15)),
        cvtransform.ChannelShuffle(p=0.1),
        cvtransform.RandomRotate90(p=0.2),
        cvtransform.RandomHorizontalFlip(p=0.5),
        cvtransform.RandomVerticalFlip(p=0.5),
        cvtransform.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
        cvtransform.ToTensor(),
        cvtransform.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=True)    #注意同步更改stage2的预处理
    ])



#=====stage 2 config
stage2_images_per_classes=7
stage2_classes_per_minibatch=7
# stage2_numofstep=0
stage2_numofstep=50000
stage2_nums_addnegative=0
stage2_size=stage1_size
stage2_pre_process_train= torchvision.transforms.Compose([
        cvtransform.RescalePad(output_size=stage2_size),
        cvtransform.ShiftScaleRotate(p=0.3,shift_limit=0.1,scale_limit=(-0.5,0.2),rotate_limit=15),
        cvtransform.IAAPerspective(p=0.1,scale=(0.05, 0.15)),
        cvtransform.ChannelShuffle(p=0.1),
        cvtransform.RandomRotate90(p=0.2),
        cvtransform.RandomHorizontalFlip(p=0.5),
        cvtransform.RandomVerticalFlip(p=0.5),
        cvtransform.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
        cvtransform.ToTensor(),
        cvtransform.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=True)
    ])



#=====extraction_config
minibatchsize_embedding=64
extraction_size=stage2_size



pre_process_test_embedding= torchvision.transforms.Compose([
        denoise_transform.Label_dependent_switcher([denoise_transform.do_nothing(),   #label为0，没有噪声
                                                    denoise_transform.de_MedianBlur(size=5), #label为1，椒盐噪声
                                                    denoise_transform.de_bilateralFilter(sigmaSpace=12)]), #label为2，高斯噪声，后面还可呀再加
        cvtransform.RescalePad(output_size=extraction_size),
        cvtransform.ToTensor(),
        cvtransform.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=False)
    ])





#学习率
def getlr(epoch,step):
    if step<=stage1_numofstep:
        lr_based=0.0004
        lr=lr_based*(0.9999503585)**step
    else:
        lr_based=0.0004*1.5
        lr=lr_based*(0.9999503585)**step
    return lr
optimizer=torch.optim.Adam


# model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        n_class=3097

        # backbone = densenet169_ibn_a(pretrained=True)
        backbone = torchvision.models.densenet169(pretrained=True)
        split_backbone=list(list(backbone.children())[0])

        model_forzen_part=split_backbone[:8]
        model_trainable_part=split_backbone[8:]

        froze_all_param(nn.Sequential(*model_forzen_part))
        backbone_before_nonlocal = nn.Sequential(*model_forzen_part,*model_trainable_part[:1])
        backbone_after_nonlocal = nn.Sequential(*model_trainable_part[1:])

        my_RGA_Module=RGA_Module(1280,44*44)

        self.backbone = nn.Sequential(backbone_before_nonlocal,\
                                        my_RGA_Module,backbone_after_nonlocal)

        self.aggregator=GeneralizedMeanPoolingP()

        self.head= BNneckHead(1664, n_class)
        self.amlinear = AMLinear_dropout(1664, n_class, m=0.35,dropout_rate=0.3)
        

        self.criterion2 = nn.CrossEntropyLoss()
        self.criterion1 = TripletLoss()


    def input2embedding(self,x): #必须有，用于提取embedding
        x=self.backbone(x)

        embedding=self.aggregator(x)

        return embedding


    def forward(self,inputs,labels=None,extract_embedding=False):

        embedding = self.input2embedding(inputs)

        if extract_embedding:
            embedding = torch.flatten(embedding, 1)
            return embedding
        out=self.head(embedding)
        logit,logit_nomargin = self.amlinear(out,labels.long())


        embedding = torch.flatten(embedding, 1)
        loss1 = self.cal_loss1(embedding,labels)
        loss2,acc = self.cal_loss2(logit,logit_nomargin,labels.long())
        return loss1,loss2,acc


    def cal_loss1(self,outputs,y):

        loss = self.criterion1(outputs, y)

        return loss
        
    def cal_loss2(self,logit,logit_nomargin,y):

        loss = self.criterion2(logit, y)
        pred = logit_nomargin.max(-1)[1]
        correct = (pred == y).sum().float()
        acc = correct / y.shape[0]
        return loss,acc

class TripletLoss(nn.Module):
    def __init__(self, margin=0.6):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)  # 获得一个简单的距离triplet函数
 
    def forward(self, inputs, labels):
 
        n = inputs.size(0)  # 获取batch_size

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)  # 每个数平方后， 进行加和（通过keepdim保持2维），再扩展成nxn维
        dist = dist + dist.t()  # 这样每个dis[i][j]代表的是第i个特征与第j个特征的平方的和
        dist.addmm_(1, -2, inputs, inputs.t())  # 然后减去2倍的 第i个特征*第j个特征 从而通过完全平方式得到 (a-b)^2
        dist = dist.clamp(min=1e-12).sqrt()  # 然后开方

        mask = labels.expand(n, n).eq(labels.expand(n, n).t())  # 这里dist[i][j] = 1代表i和j的label相同， =0代表i和j的label不相同
        dist_ap, dist_an = [], []
        for i in range(n):
            if len(dist[i][mask[i]])>0 and len(dist[i][mask[i] == 0])>0:
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # 在i与所有有相同label的j的距离中找一个最大的
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # 在i与所有不同label的j的距离找一个最小的
        dist_ap = torch.cat(dist_ap)  # 将list里的tensor拼接成新的tensor
        dist_an = torch.cat(dist_an)
 
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)  # 声明一个与dist_an相同shape的全1tensor

        loss = self.ranking_loss(dist_an, dist_ap, y)


        return loss
if __name__ == '__main__':



    mynn=Model()
    test_x=torch.rand((3,3,512,512))
    test_label=torch.ones(3)
    b=mynn(test_x,test_label)
