###
 # @Author: your name
 # @Date: 2020-08-11 05:01:22
 # @LastEditTime: 2020-08-14 01:36:28
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /HW2/tensorboardsummary.sh
### 
#!/bin/bash
#victor 2019.1.12

#sample: ./trainallmutation.sh

modelsdir=models
expsdir=exp
mergedir=log/mergedtfbrecord


mergedir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $mergedir ${PWD}`
modelsdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $modelsdir ${PWD}`
expsdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $expsdir ${PWD}`


mkdir -p $mergedir

for modeldir in ${modelsdir}/*
do  

if [ -d $modeldir ] 
then



modelname=`basename $modeldir`


if [ ! -d $mergedir/$modelname ];then
ln -s $modeldir/log $mergedir/$modelname


fi


fi



done


for modeldir in ${expsdir}/*
do  

if [ -d $modeldir ] 
then



modelname=`basename $modeldir`


if [ ! -d $mergedir/$modelname ];then
ln -s $modeldir/log $mergedir/$modelname


fi


fi



done


tensorboard --logdir=$mergedir --port=6006 --bind_all












