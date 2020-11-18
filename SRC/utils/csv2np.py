import pandas as pd
import numpy as np


def csv2np_test(csv_path):
    df = pd.read_csv(csv_path, header=None)
    num_of_images = 10
    embedding_dim = df.shape[1] - 1
    embeddings = np.zeros((num_of_images, embedding_dim))

    labels = [0] * num_of_images
    for i in range(num_of_images):
        embeddings[i, :] = df.iloc[i, 1:].astype(np.float32)
        labels[i] = df[0][i]

    return embeddings, labels


def csv2np(csv_path):

    df = pd.read_csv(csv_path, header=None)
    num_of_images = df.shape[0]
    embedding_dim = df.shape[1]-1
    embeddings=np.zeros((num_of_images,embedding_dim))

    labels=[0]*num_of_images
    for i in range(num_of_images):
        embeddings[i,:]=df.iloc[i, 1:].astype(np.float32)
        labels[i]=df[0][i]

    return embeddings,labels


if __name__ == '__main__':
    a,b=csv2np("/home/yufei/HUW/models/baseline/embeddings/resnet50/query.csv")