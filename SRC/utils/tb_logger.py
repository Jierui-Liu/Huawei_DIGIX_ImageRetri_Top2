'''
Author: your name
Date: 2020-08-11 05:03:11
LastEditTime: 2020-08-14 23:33:52
LastEditors: Please set LastEditors
Description: In User Settings Edit  
FilePath: /HW2/SRC/utils/tb_logger.py
'''

import cv2
import torch
from tensorboardX import SummaryWriter
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
class fulldata_CE_logger(object):
    def __init__(self,logger_dir):
        self.log_writer=SummaryWriter(log_dir=logger_dir, comment='', purge_step=None, max_queue=10, flush_secs=20, filename_suffix='')
    def add_log(self,step,train_loss,test_loss,train_acc,test_acc,lr=0):
        print("step", step, "train_loss", train_loss, "train_acc", train_acc)
        self.log_writer.add_scalar('FULL_DATA/Lr', lr, step)
        self.log_writer.add_scalar('FULL_DATA/train_loss', train_loss, step)
        self.log_writer.add_scalar('FULL_DATA/train_acc', train_acc, step)




class test_CE_logger(object):
    def __init__(self,logger_dir):
        self.log_writer=SummaryWriter(log_dir=logger_dir, comment='', purge_step=None, max_queue=10, flush_secs=20, filename_suffix='')
    def add_log(self, step, train_loss , train_acc, top_1_acc, mAP10, lr=0):
        print("step", step, "train_loss", train_loss, "train_acc", train_acc,"top1_acc", top_1_acc, "mAP10", mAP10)
        self.log_writer.add_scalar('TEST/Lr', lr, step)
        self.log_writer.add_scalar('TEST/train_loss', train_loss, step)
        self.log_writer.add_scalar('TEST/top_1_acc', top_1_acc, step)
        self.log_writer.add_scalar('TEST/train', train_acc, step)
        self.log_writer.add_scalar('TEST/mAP10', mAP10, step)
        self.log_writer.add_scalar('TEST/(top_1_acc+mAP10)/2', (mAP10+top_1_acc)/2, step)



class trip_ce_loss_logger(object):
    def __init__(self,logger_dir):
        self.log_writer=SummaryWriter(log_dir=logger_dir, comment='', purge_step=None, max_queue=10, flush_secs=20, filename_suffix='')
    def add_log(self,step,train_loss_tri,train_loss_ce,train_loss,train_acc,lr=0):
        #jerry 自己写
        self.log_writer.add_scalar('Lr', lr, step)
        self.log_writer.add_scalar('Loss/train_loss_tri', train_loss_tri, step)
        self.log_writer.add_scalar('Loss/train_loss_ce', train_loss_ce, step)
        self.log_writer.add_scalar('Loss/train_loss', train_loss, step)
        self.log_writer.add_scalar('Acc/train_acc', train_acc, step)

        
class trip_ce_mod_loss_logger(object):
    def __init__(self,logger_dir):
        self.log_writer=SummaryWriter(log_dir=logger_dir, comment='', purge_step=None, max_queue=10, flush_secs=20, filename_suffix='')
    def add_log(self,step,train_loss_tri,train_loss_ce,train_loss_mod,train_loss,train_acc,lr=0):
        #jerry 自己写
        self.log_writer.add_scalar('Lr', lr, step)
        self.log_writer.add_scalar('Loss/train_loss_tri', train_loss_tri, step)
        self.log_writer.add_scalar('Loss/train_loss_ce', train_loss_ce, step)
        self.log_writer.add_scalar('Loss/train_loss_mod', train_loss_mod, step)
        self.log_writer.add_scalar('Loss/train_loss', train_loss, step)
        self.log_writer.add_scalar('Acc/train_acc', train_acc, step)

        
class trip_ce_loss_logger_addembedding(object):
    def __init__(self,logger_dir):
        self.log_writer=SummaryWriter(log_dir=logger_dir, comment='', purge_step=None, max_queue=10, flush_secs=20, filename_suffix='')
    def add_log(self,step,train_loss_tri,train_loss_ce,train_loss,train_acc,lr=0,features=None, metadata=None):
        #jerry 自己写
        self.log_writer.add_scalar('Lr', lr, step)
        self.log_writer.add_scalar('Loss/train_loss_tri', train_loss_tri, step)
        self.log_writer.add_scalar('Loss/train_loss_ce', train_loss_ce, step)
        self.log_writer.add_scalar('Loss/train_loss', train_loss, step)
        self.log_writer.add_scalar('Acc/train_acc', train_acc, step)

    def addembedding_log(self,step,features=None, metadata=None):
        #jerry 自己写
        self.log_writer.add_embedding(features, metadata=metadata,global_step=step,tag='embedding')


class vis_attention(object):
    def __init__(self,logger_dir):
        self.log_writer = SummaryWriter(log_dir=logger_dir, comment='', purge_step=None, max_queue=10, flush_secs=20,
                                        filename_suffix='')
    def add_log(self,img,attention_map,f_name,step=0):
        #img是0-255范围
        #attention_map是0-1范围
        attention_map=attention_map.detach().numpy()
        self.log_writer.add_image(f_name, img/255, global_step=step,dataformats='CHW',)
        self.log_writer.add_image(f_name+"_attention_map", attention_map/attention_map.max(), global_step=step, dataformats='HW')
        h=img.shape[1]
        w=img.shape[2]
        attention_map=cv2.resize(attention_map,dsize=(h,w))

        self.log_writer.add_image(f_name+"_mix", img*(attention_map-attention_map.min())/attention_map.max()/255, global_step=step, dataformats='CHW')
