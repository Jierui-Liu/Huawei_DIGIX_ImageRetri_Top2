###
 # @Author: your name
 # @Date: 2020-08-11 05:03:16
 # @LastEditTime: 2020-08-23 13:54:37
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /HW2/trainmodel.sh
### 
#!/bin/bash
#victor 2019.1.12

#example: ./trainmodel.sh ./models/dnn_cvmn_cqcc/mutation/dnn_bn_dropout_1a.py



str=$1





num_of_gpu=$(grep "numofGPU=" $str | tr -cd "[0-9]") 



array=(${str//// })



lenarray=${#array[*]}
mutation=${array[lenarray-1]}
mutarray=(${mutation//./ })
mutation=${mutarray[0]}

modelname=${array[lenarray-3]}

for i in $(seq 0 $[$lenarray-3])
do

modeldir=$modeldir${array[$i]}/
done




#queue.pl 启动训练
jobname=r_$modelname"_"$mutation

if [ ! -n "$2" ]; then
  QUEUE_NAME=GPU_QUEUE@@amax2017
else
  QUEUE_NAME=GPU_QUEUE@compute-0-$2.local
fi

ifresume_training=false

if [ -d $modeldir/newest_model_saved/$mutation ];then

echo "resume training(y/n)"
read dec


if [ $dec == "y" ]


then
echo "resume training"
ifresume_training=true

else
echo "new training"
ifresume_training=false


fi


fi








if [ $HOSTNAME == "nghci-Amax-2017" ]
then

queue.pl -q $QUEUE_NAME -N $jobname --num-threads $num_of_gpu log/$modelname/$jobname.log $modeldir"run.sh" $mutation $ifresume_training &

queue_pid=$!

echo "begin to run "$modelname "mutation:"$mutation with $num_of_gpu GPU
echo "submited to" $QUEUE_NAME


else 


$modeldir"run.sh" $mutation $ifresume_training


fi






