

def chage_lr_to(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr