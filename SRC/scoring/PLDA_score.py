
import sys

sys.path.append("/home/yufei/pytorch-project/asv_weibing/models/plda_while_training")
sys.path.append("/home/yufei/pytorch-project/asv_weibing")
sys.path.append("/home/yufei/HUW")
import torch
from mutation.baseline import torch_PLDA
from SRC.utils.csv2np import csv2np as csv2np



embedding_dir=sys.argv[1]


train,labels = csv2np(embedding_dir+'/train.csv')





PLDA=torch_PLDA(margin=0,plda_dim=200)
PLDA.train_statistic_PLDA(train,labels)
#PLDA.save_to_file(embedding_dir+"/PLDA_MODEL")


gallery, gallery_fname_all =csv2np(embedding_dir + '/gallery.csv')
query, query_fname_all = csv2np(embedding_dir + '/query.csv')

gallery=torch.tensor(torch.from_numpy(gallery), dtype=torch.float32)
query=torch.tensor(torch.from_numpy(query), dtype=torch.float32)


result_file_path=embedding_dir+'/final.txt'
fp = open(result_file_path, 'w', encoding='utf-8')
with torch.no_grad():
    for i in range(query.shape[0]):
        query_name=query_fname_all[i]
        input=query[i,:].reshape(1,-1)
        score=PLDA.PLDA_score(input,gallery)

        sorted_score_index=score.numpy().argsort()[-10:][::-1]
        res_tmp='{},{},{},{},{},{},{},{},{},{}'.format\
            (gallery_fname_all[sorted_score_index[0]],\
            gallery_fname_all[sorted_score_index[1]],\
            gallery_fname_all[sorted_score_index[2]],\
            gallery_fname_all[sorted_score_index[3]],\
            gallery_fname_all[sorted_score_index[4]],\
            gallery_fname_all[sorted_score_index[5]],\
            gallery_fname_all[sorted_score_index[6]],\
            gallery_fname_all[sorted_score_index[7]],\
            gallery_fname_all[sorted_score_index[8]],\
            gallery_fname_all[sorted_score_index[9]],\
             )
        res_tmp=query_name+',{'+res_tmp+'}'
        fp.write(res_tmp)
        fp.write('\n')


