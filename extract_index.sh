#!/bin/bash
#victor 2019.1.12

#example: ./extractmodel.sh ./models/trip_loss/mutation/resnet50.py



str=$1

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





# if [ -d $modeldir/embeddings/$mutation ];then

# echo "embedding already extracted"
# exit 0;


# fi



echo "begin to extract "$modelname "mutation:"$mutation

#queue.pl 启动训练
jobname=e_i_$modelname"_"$mutation


if [ ! -n "$2" ]; then
  QUEUE_NAME=GPU_QUEUE@@amax2017
else
  QUEUE_NAME=GPU_QUEUE@compute-0-$2.local
fi


if [ $HOSTNAME == "nghci-Amax-2017" ]
then

queue.pl -q $QUEUE_NAME -N $jobname --num-threads 2 log/$modelname/$jobname.log $modeldir"run_multi.sh" $mutation &


queuepid=$!

echo "submited to" $QUEUE_NAME

exit 0;

else 


$modeldir"run_multi.sh" $mutation
fi








