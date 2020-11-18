###
 # @Author      : now more
 # @Contact     : lin.honghui@qq.com
 # @LastEditors: Please set LastEditors
 # @Description : 
 # @LastEditTime: 2020-11-18 16:03:23
### 
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/home/LinHonghui/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/home/LinHonghui/anaconda3/etc/profile.d/conda.sh" ]; then
#         . "/home/LinHonghui/anaconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/home/LinHonghui/anaconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# <<< conda initialize <<<

cd tools
# conda activate freihand


# config_file="../configs/testC_noise_efficientb3_patch640.py"
# load_path="../checkpoints/efficient_b3_e300/testC_efficientb3_300.pth"


config_file="../configs/testC_efficientb4_patch640.py"
load_path="../checkpoints/efficient_b4_ori/testC_efficientb4_220.pth"

# config_file="../configs/testC_noise_efficientb5_patch640.py"
# load_path="../checkpoints/efficient_b5_e80/testC_efficientb5_80.pth"



python extract_features.py    -config_file $config_file -load_path $load_path -max_num_devices 2
