
import sys
sys.path.append(".")
from SRC.utils.get_gpu import get_gpu
from torch.utils.data import DataLoader
from SRC.datasets.triple_addnegative_dataset_unique import train_dataset
from SRC.utils.tb_logger import trip_ce_loss_logger
from SRC.During_training.change_lr import chage_lr_to
from SRC.During_training.eval_model import *
from SRC.During_training.load_model import load_model_from_cpd
import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import time 
from apex import amp
#训练的时候加噪声
#测试也不加噪声




def my_collect_fn(batch_list): # train_collect_fn
    #注意这个collect_fn只能用于triloss的dataloader，（batchsize=1的时候）

    image,labels = batch_list[0]
    return image,labels



mutationname=sys.argv[1]

mynn = __import__('mutation.' + mutationname, fromlist=True)

modeldir = sys.path[0]

temp = modeldir.split('/')
modelname = temp[-1]

newest_model_savepath = modeldir + '/newest_model_saved/' + mutationname+'.pth'

loggerdir = modeldir + '/log/' + mutationname
checkpoint_dir=modeldir + '/newest_model_saved/' + mutationname
resume_training=int(sys.argv[2])

logger=trip_ce_loss_logger(loggerdir)





Model=mynn.Model()
step=0
if resume_training:
    step=load_model_from_cpd(Model,checkpoint_dir)
    print("resuming from step:",step)
# Model.load_state_dict({k.replace('module.',''):v for k, v  in torch.load('/home/LiuJierui/Proj/HUW7/models/test_B/newest_model_saved/densenet169_RAG_704.pth', map_location='cpu').items()})


gpu_list=get_gpu(mynn.numofGPU)
if len(gpu_list)==0:
    sys.exit(1)
# Model = torch.nn.DataParallel(Model, device_ids=gpu_list).cuda()
Model = Model.cuda()



stage1_tr=train_dataset(mynn.data_dir,\
                images_per_classes=mynn.stage1_images_per_classes,\
                classes_per_minibatch=mynn.stage1_classes_per_minibatch,\
                nums_addnegative=mynn.stage1_nums_addnegative,\
                transform=mynn.stage1_pre_process_train)


train_loader = DataLoader(dataset=stage1_tr, batch_size=1, shuffle=True ,num_workers=8,collate_fn=my_collect_fn)
test_loader = DataLoader(dataset=stage1_tr, batch_size=1, shuffle=True ,num_workers=4,collate_fn=my_collect_fn)



optimizer=mynn.optimizer(Model.parameters(),lr=mynn.getlr(0,0))




epoch=0
steptoend=False
# 在训练最开始之前实例化一个GradScaler对象
# scaler = GradScaler()
Model, optimizer = amp.initialize(Model, optimizer, opt_level="O1")

#stage1===========
while True:

    str_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print("{},{} epoch{}".format(str_time,mutationname,epoch))
    chage_lr_to(optimizer,mynn.getlr(epoch,step))

    for im, label in train_loader:
        if step%500==0:
            print_train_loss_tri,print_train_loss_ce,print_train_acc=\
                        eval_loss_acc_tri(Model,test_loader,100)
            print_train_loss=print_train_loss_tri+print_train_loss_ce/mynn.ce_tri_ratio
            print(print_train_loss_tri,print_train_loss_ce,print_train_loss,print_train_acc)
            logger.add_log(step,print_train_loss_tri,print_train_loss_ce,print_train_loss,print_train_acc,optimizer.param_groups[0]['lr'])
        if(step%10000==0):
            torch.save(Model.state_dict(), modeldir + '/newest_model_saved/{}/s{}_'.format(mutationname,step) + mutationname+'.pth',_use_new_zipfile_serialization=False)

        if step>=mynn.stage1_numofstep:
            steptoend=True
            break
        im=im.cuda()
        label=label.cuda()
        # im=im[:-4].cuda()
        # label=label[:-4].cuda()  

        train_loss_tri,train_loss_ce,train_acc=Model(im,labels=label)
        train_loss_tri=train_loss_tri.mean()
        train_loss_ce=train_loss_ce.mean()
        train_acc=train_acc.mean()
        #trip+ce
        train_loss = train_loss_tri+train_loss_ce/mynn.ce_tri_ratio

        # # 前向过程(model + loss)开启 autocast
        # with autocast():
        #     # output = model(input)
        #     # loss = loss_fn(output, target)
        #     train_loss_tri,train_loss_ce,train_acc=Model(im,labels=label)

        #     train_loss_tri=train_loss_tri.mean()
        #     train_loss_ce=train_loss_ce.mean()
        #     train_acc=train_acc.mean()
        #     #trip+ce
        #     train_loss = train_loss_tri+train_loss_ce/mynn.ce_tri_ratio
        # print(train_loss.item())

        optimizer.zero_grad()
        with amp.scale_loss(train_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # train_loss.backward()
        optimizer.step()

        # # Scales loss. 为了梯度放大.
        # scaler.scale(train_loss).backward()
        # # scaler.step() 首先把梯度的值unscale回来.
        # # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
        # # 否则，忽略step调用，从而保证权重不更新（不被破坏）
        # scaler.step(optimizer)
        # # 准备着，看是否要增大scaler
        # scaler.update()



        step=step+1
    epoch = epoch + 1
    train_loader.dataset.shuffle()
    torch.save(Model.state_dict(), newest_model_savepath)
    if steptoend:
        break

del train_loader
del test_loader


stage2_tr=train_dataset(mynn.data_dir,\
                images_per_classes=mynn.stage2_images_per_classes,\
                classes_per_minibatch=mynn.stage2_classes_per_minibatch,\
                nums_addnegative=mynn.stage2_nums_addnegative,\
                transform=mynn.stage2_pre_process_train)


train_loader = DataLoader(dataset=stage2_tr, batch_size=1, shuffle=True ,num_workers=8,collate_fn=my_collect_fn)
test_loader = DataLoader(dataset=stage2_tr, batch_size=1, shuffle=True ,num_workers=4,collate_fn=my_collect_fn)

steptoend=False

#stage2===========
while True:

    str_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print("{},{} epoch{}".format(str_time,mutationname,epoch))
    chage_lr_to(optimizer,mynn.getlr(epoch,step))

    for im, label in train_loader:
        if step%500==0:
            print_train_loss_tri,print_train_loss_ce,print_train_acc=\
                        eval_loss_acc_tri(Model,test_loader,20)
            print_train_loss=print_train_loss_tri+print_train_loss_ce/mynn.ce_tri_ratio
            logger.add_log(step,print_train_loss_tri,print_train_loss_ce,print_train_loss,print_train_acc,optimizer.param_groups[0]['lr'])
        if(step%10000==0):
            torch.save(Model.state_dict(), modeldir + '/newest_model_saved/{}/s{}_'.format(mutationname,step) + mutationname+'.pth',_use_new_zipfile_serialization=False)

        if step>=mynn.stage1_numofstep+mynn.stage2_numofstep:
            steptoend=True
            break

        im=im.cuda()
        label=label.cuda()
        # im=im[:-4].cuda()
        # label=label[:-4].cuda() 

        train_loss_tri,train_loss_ce,train_acc=Model(im,labels=label)
        train_loss_tri=train_loss_tri.mean()
        train_loss_ce=train_loss_ce.mean()
        train_acc=train_acc.mean()
        #trip+ce
        train_loss = train_loss_tri+train_loss_ce/mynn.ce_tri_ratio

        # # 前向过程(model + loss)开启 autocast
        # with autocast():
        #     # output = model(input)
        #     # loss = loss_fn(output, target)
        #     train_loss_tri,train_loss_ce,train_acc=Model(im,labels=label)

        #     train_loss_tri=train_loss_tri.mean()
        #     train_loss_ce=train_loss_ce.mean()
        #     train_acc=train_acc.mean()
        #     #trip+ce
        #     train_loss = train_loss_tri+train_loss_ce/mynn.ce_tri_ratio

        optimizer.zero_grad()
        # train_loss.backward()
        with amp.scale_loss(train_loss, optimizer) as scaled_loss:
            # scaled_loss.backward(retain_graph=True)
            scaled_loss.backward()
        optimizer.step()

        # # Scales loss. 为了梯度放大.
        # scaler.scale(train_loss).backward()
        # # scaler.step() 首先把梯度的值unscale回来.
        # # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
        # # 否则，忽略step调用，从而保证权重不更新（不被破坏）
        # scaler.step(optimizer)
        # # 准备着，看是否要增大scaler
        # scaler.update()


        step=step+1
    epoch = epoch + 1
    train_loader.dataset.shuffle()
    torch.save(Model.state_dict(), newest_model_savepath,_use_new_zipfile_serialization=False)
    if steptoend:
        break