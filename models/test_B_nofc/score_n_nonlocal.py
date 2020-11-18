'''
Author: your name
Date: 2020-08-18 21:36:18
LastEditTime: 2020-08-20 09:51:10
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /HW2/score_models/baseline/score.py
'''

import sys
sys.path.append(".")
import os
from SRC.index.index import *
import random
import shutil



mutationname =sys.argv[1]  #baseline

mynn = __import__('mutation.' + mutationname, fromlist=True)

modeldir = sys.path[0]

temp = modeldir.split('/')
modelname = temp[-1]

submission_nonlocal_file_path = modeldir + '/submission_file/' + mutationname+"_nonlocal_submission.csv"
submission_rag_file_path = modeldir + '/submission_file/' + mutationname+"_rag_submission.csv"
submission_attention_file_path = modeldir + '/submission_file/' + mutationname+"_attention_submission.csv"



temp_dir=os.path.join(modeldir,'temp',mutationname)


try:
    os.mkdir(temp_dir)
except:
    pass
yaml_file=open(os.path.join(temp_dir,"config.yaml"),mode='w')
yaml_config=mynn.yaml_config.replace("$temp_dir_UNiQUE_mark",temp_dir)
yaml_file.write(yaml_config)
yaml_file.close()
#trans_json_to here   #把临时文件放在temp_dir
trans_json_to_here(here_dir=temp_dir,\
                json_dir=mynn.embeddings_dir_nonlocal)
#index
index_n(config_file=os.path.join(temp_dir,"config.yaml"),\
            save_file=submission_nonlocal_file_path)
shutil.rmtree(temp_dir)

        







