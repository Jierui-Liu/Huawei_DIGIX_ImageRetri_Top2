###
 # @Author      : now more
 # @Contact     : lin.honghui@qq.com
 # @LastEditors: Please set LastEditors
 # @Description : 
 # @LastEditTime: 2020-11-18 16:00:21
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
# # <<< conda initialize <<<

conda activate pytorch


python ./index_tools/trans_json_to_here.py \
--here_dir ./features/hw2_json \
--json_dir /home/LinHonghui/Project/v2/features/exp/20201102_testC_efficientb5_80



python ./index_tools/index.py \
        -cfg ./index_configs/hw_json.yaml \
        -sf ./result_tmp/submission.csv


