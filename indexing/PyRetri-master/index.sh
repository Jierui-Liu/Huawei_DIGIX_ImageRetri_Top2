#!/bin/bash




for((attemp=1;attemp<=3;attemp++));  
do   

. $(dirname $0)/getgpu.sh 1
# /home/yufei/.conda/envs/pytorch_gpu/bin/python test_gpu.py
 #改成你要使用GPU的python
# /home/yufei/.conda/envs/pytorch_hw/bin/python main/single_index.py -cfg configs/oxford.yaml

/home/yufei/.conda/envs/pytorch_hw/bin/python main/index.py -cfg configs/oxford.yaml 








exitcode=$?
echo $exitcode

if [ $exitcode -eq 0 ];then
normexit=1
break
fi
done  


if [ $normexit -eq 1 ];then   

exit 0

else

exit 1
fi















