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








$modeldir"copy_nofc.sh" $mutation








