import torch
import os
print(os.environ.get("CUDA_VISIBLE_DEVICES"))
print('torch.cuda.is_available:',torch.cuda.is_available())