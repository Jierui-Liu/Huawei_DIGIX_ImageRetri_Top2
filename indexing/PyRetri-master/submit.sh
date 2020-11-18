#!/bin/bash
#victor 2019.1.12

#example: ./trainmodel.sh ./models/dnn_cvmn_cqcc/mutation/dnn_bn_dropout_1a.py


# QUEUE_NAME=GPU_QUEUE@compute-0-7.local #GPU计算节点，需要GPU的用这个
QUEUE_NAME=CPU_QUEUE  #cpu计算节点，不需要GPU的用这个

jobname=hw_j

log_path=log/hw_j.log

# queue.pl -q $QUEUE_NAME -N $jobname $log_path ./make_data_json.sh
# queue.pl -q $QUEUE_NAME -N $jobname $log_path ./extract_feature.sh
# queue.pl -q $QUEUE_NAME -N $jobname $log_path ./index.sh

# queue.pl -q $QUEUE_NAME -N $jobname $log_path ./my_make_data_json.sh
# queue.pl -q $QUEUE_NAME -N $jobname $log_path ./extract_feature.sh
queue.pl -q $QUEUE_NAME -N $jobname $log_path ./index.sh



# qstat #查询正在运行的任务


 
# qdel  JOB-ID 别弄错ID，






