import torch


aa=torch.load("/home/LinHonghui/HW2/models/baseline_1/newest_model_saved/baseline_attention_early.pth", map_location='cpu')
bb=torch.load("/home/LinHonghui/HW2/models/baseline_1/newest_model_saved/baseline_singleGPU_eula0_6_ce2_4_7_512_enhance.pth", map_location='cpu')



c=1