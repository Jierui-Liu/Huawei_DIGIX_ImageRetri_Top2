#!/bin/bash




for((attemp=1;attemp<=8;attemp++));  
do   

. $(dirname $0)/getgpu.sh 1

 #改成你要使用GPU的python
/home/yufei/.conda/envs/pytorch_gpu/bin/python my_make_data_json.py 




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








