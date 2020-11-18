



import os
import torch


def get_step_from_checkpoint(fname):
    return int(fname.split("_")[0][1:])

def load_model_from_cpd(Model,chekpoint_dir):
    checkpoints=os.listdir(chekpoint_dir)
    newest_step=0
    newest_model_fname=""
    print(checkpoints)
    for checkpoint in checkpoints:
        step=get_step_from_checkpoint(checkpoint)
        if step>newest_step:
            newest_step=step
            newest_model_fname=checkpoint
    newest_model_savepath=os.path.join(chekpoint_dir,newest_model_fname)
    Model.load_state_dict({k.replace('module.',''):v for k, v  in torch.load(newest_model_savepath, map_location='cpu').items()})

    return newest_step

def load_model_from_cp_nofc(Model,chekpoint):
    chekpoint_dict={k.replace('module.',''):v for k, v  in torch.load(chekpoint, map_location='cpu').items()}
    # 将原始网络结构与修改后网络结构相同的键值对放到一个有序字典当中，不相同的键值对则被删除
    error=[1 for k in Model.state_dict() if k not in chekpoint_dict]
    pretrained_dict = {k: v for k, v in chekpoint_dict.items() if k in Model.state_dict()}
    error_1=[1 for k in pretrained_dict if k not in Model.state_dict()]
    if len(error)!=0 or len(error_1)!=0:
            print("=====================Something wrong when saving model without fc=====================")
            exit(0)
    # 将这个有序字典传给load_state_dict
    Model.load_state_dict(pretrained_dict)
    # print(pretrained_dict.keys())


if __name__ == '__main__':
    cpd="/home/yufei/HUW4/models/test_A/newest_model_saved/densenet169_nonlocal_dot_amsoftmax_dropout"
    load_model_from_cpd([],cpd)


