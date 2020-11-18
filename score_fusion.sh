###
 # @Author: your name
 # @Date: 2020-08-18 21:57:12
 # @LastEditTime: 2020-08-19 09:13:02
 # @LastEditors: your name
 # @Description: In User Settings Edit
 # @FilePath: /HW2/score.sh
### 

#!/bin/bash
#victor 2019.1.12


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



#queue.pl 启动训练
jobname=score_$modelname"_"$mutation



$modeldir"score.sh" $mutation





