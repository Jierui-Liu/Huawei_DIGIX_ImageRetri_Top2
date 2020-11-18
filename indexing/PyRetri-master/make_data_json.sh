#!/bin/bash




for((attemp=1;attemp<=3;attemp++));  
do   

. $(dirname $0)/getgpu.sh 1

 #改成你要使用GPU的python
/home/yufei/.conda/envs/pytorch_gpu/bin/python main/make_data_json.py \
 -d data/cbir/oxford/gallery/ -sp data_jsons/oxford_gallery.json -t oxford -gt data/cbir/oxford/gt

/home/yufei/.conda/envs/pytorch_gpu/bin/python main/make_data_json.py \
 -d data/cbir/oxford/query/ -sp data_jsons/oxford_query.json -t oxford -gt data/cbir/oxford/gt






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








