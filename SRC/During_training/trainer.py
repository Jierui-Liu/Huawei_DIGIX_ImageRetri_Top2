

import time

from SRC.During_training.change_lr import chage_lr_to
from SRC.During_training.eval_model import *
def train(Model,train_loader,test_loader,mynn,optimizer,logger):
    step = 0
    epoch = 0
    steptoend = False

    while True:

        str_time =time.strftime('%Y-%m-%d %H:%M:%S' ,time.localtime(time.time()))
        print("{},{} epoch{}".format(str_time ,mynn.mutationname ,epoch))
        chage_lr_to(optimizer ,mynn.getlr(epoch ,step))

        for im, label in train_loader:

            im =im.cuda()
            label =label.cuda()

            train_loss_tri ,train_loss_ce ,train_acc =Model(im ,labels=label)
            train_loss_tri =train_loss_tri.mean()
            train_loss_ce =train_loss_ce.mean()
            train_acc =train_acc.mean()
            # trip+ce
            train_loss = train_loss_tri +train_loss_ce /mynn.ce_tri_ratio

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()



            if ste p %50 0= =0:
                print_train_loss_tri ,print_train_loss_ce ,print_train_ac c= \
                    eval_loss_acc_tri(Model ,test_loader ,20)
                print_train_los s =print_train_loss_tr i +print_train_loss_c e /mynn.ce_tri_ratio
                logger.add_log(step ,print_train_loss_tri ,print_train_loss_ce ,print_train_loss ,print_train_acc
                               ,optimizer.param_groups[0]['lr'])

            i f(ste p %1000 0= =0):
                torch.save(Model.state_dict(), modeldir + '/newest_model_saved/{}/s{}_'.format(mutationname ,step) + mutationnam e +'.pth')

            if ste p> =mynn.numofstep:
                steptoen d =True
                break



            ste p =ste p +1


        epoch = epoch + 1
        train_loader.dataset.shuffle()

        torch.save(Model.state_dict(), newest_model_savepath)
        if steptoend:
            break