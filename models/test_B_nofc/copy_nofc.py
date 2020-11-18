
import sys
sys.path.append(".")
from SRC.utils.get_gpu import get_gpu
from torch.utils.data import DataLoader
from SRC.datasets.triple_addnegative_dataset_unique import train_dataset
from SRC.utils.tb_logger import trip_ce_loss_logger
from SRC.During_training.change_lr import chage_lr_to
from SRC.During_training.eval_model import *
from SRC.During_training.load_model import load_model_from_cp_nofc
import torch

import time
#训练的时候不加噪声
#测试也不加噪声







mutationname=sys.argv[1]

mynn = __import__('mutation.' + mutationname, fromlist=True)

modeldir = sys.path[0]

temp = modeldir.split('/')
modelname = temp[-1]

checkpoint = modeldir[:-5] + '/newest_model_saved/' + mutationname+'.pth'
newest_model_savepath = modeldir + '/newest_model_saved/' + mutationname+'.pth'
print("checkpoint",checkpoint)
print("newest_model_savepath",newest_model_savepath)


Model=mynn.Model()
load_model_from_cp_nofc(Model,checkpoint)
torch.save(Model.state_dict(), newest_model_savepath)
print('finished')