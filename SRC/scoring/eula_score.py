import numpy as np


def eula_score(query_embeddings, gallery_embeddings):
    # embedding是np数组（样本数，embedding维度）

    numof_gallery =gallery_embeddings.shape[0]
    numof_query=query_embeddings.shape[0]

    gallery_powered=np.sum(np.power(gallery_embeddings, 2),axis=1)
    query_powered=np.sum(np.power(query_embeddings, 2),axis=1)
    score=np.tile(gallery_powered,(numof_query,1))+  np.tile(query_powered,(numof_gallery,1)).T

    dot_ = query_embeddings.dot(np.transpose(gallery_embeddings, (1, 0)))  #3,5

    score=np.sqrt(score-2*dot_)

    return -score  # 分数矩阵 query样本数目xgallery样本数目



if __name__ == '__main__':

    a=np.array([[1,1,1],
               [1,2,1],
               ])
    b=np.array([[1,2,3],
               [1,1,1],
               [2,2,2]
               ])
    bb=eula_score(a,b)

    c=1
