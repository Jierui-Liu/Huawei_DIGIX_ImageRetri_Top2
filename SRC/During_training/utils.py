'''
Author: your name
Date: 2020-08-11 05:02:49
LastEditTime: 2020-08-12 20:56:06
LastEditors: your name
Description: In User Settings Edit
FilePath: /HW2/SRC/During_training/utils.py
'''


def froze_all_param(partial_model):
    for child in partial_model.children():
        child.eval()
        for param in child.parameters():
            param.requires_grad = False


def froze_all_param_ufbn(partial_model):
    for child in partial_model.children():
        for param in child.parameters():
            param.requires_grad = False


def unfroze_all_param(partial_model):
    for child in partial_model.children():
        child.train()
        for param in child.parameters():
            param.requires_grad = True
#bn层可能没有冻结

