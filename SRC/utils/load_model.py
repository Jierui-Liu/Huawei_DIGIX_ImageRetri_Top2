



import os
import torch

def load_model_from_cpd(Model,chekpoint_dir):
    checkpoints=os.listdir(chekpoint_dir)
    newest_step=0
    for checkpoint in checkpoints:
        pass




    Model.load_state_dict({k.replace('module.',''):v for k, v  in torch.load(newest_model_savepath, map_location='cpu').items()})




if __name__ == '__main__':
    cpd="./models/test_A/newest_model_saved/densenet169_nonlocal_dot_amsoftmax_dropout"
    load_model_from_cpd([],cpd)


