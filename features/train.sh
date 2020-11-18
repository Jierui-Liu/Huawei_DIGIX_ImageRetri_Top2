###
 # @Author      : now more
 # @Contact     : lin.honghui@qq.com
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2020-11-18 16:02:44
 # @Description : 
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
conda activate pytorch







config_file="../configs/testC_efficientb4.py"
load_path="../checkpoints/20201027_testC_efficientb4/testC_efficientb4_150.pth"

python train.py    -config_file $config_file 
# python train.py    -config_file $config_file -load_path $load_path





