
from PIL import Image
import pandas as pd
import os
import random
from shutil import copyfile


embedding_path="/home/yufei/HUW/models/baseline/embeddings/densenet169_amsoftmax_morefrozen/final.txt"

out_dir="/home/yufei/HUW/debug"

sf = pd.read_csv(submit_file_path, header=None)




for i in range(20):
    num=random.randint(0,len(sf))

    d_dir=os.path.join(out_dir,str(i))
    os.makedirs(d_dir, exist_ok=True)

    query_img_path = os.path.join(test_data_dir, "query", sf[0][num])
    query_img_path = query_img_path.replace(".npy", ".jpg")
    copyfile(query_img_path,os.path.join(d_dir,sf[0][num]+"query.jpg"))



    for j in range(1,11):
        gallery_img_name=sf[j][num]
        gallery_img_name= gallery_img_name.replace(".npy",".jpg")
        gallery_img_name=gallery_img_name.replace(' ','')
        gallery_img_name=gallery_img_name.replace('{','')
        gallery_img_name=gallery_img_name.replace('}','')
        gallery_img_path = os.path.join(test_data_dir, "gallery",gallery_img_name )
        copyfile(gallery_img_path, os.path.join(d_dir, "gallery" + str(j) + ".jpg"))



