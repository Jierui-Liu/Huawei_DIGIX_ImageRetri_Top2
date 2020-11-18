import numpy as np



def cos_score(query_embeddings,gallery_embeddings):
    #embedding是np数组（样本数，embedding维度）
    # 归一化
    query_embeddings_norm=np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_embeddings=query_embeddings/query_embeddings_norm
    gallery_embeddings_norm=np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
    gallery_embeddings=gallery_embeddings/gallery_embeddings_norm

    # 计算距离
    gallery_embeddings_t = np.transpose(gallery_embeddings,(1, 0))
    score = query_embeddings.dot(gallery_embeddings_t)
    
    return score   #分数矩阵 query样本数目xgallery样本数目




