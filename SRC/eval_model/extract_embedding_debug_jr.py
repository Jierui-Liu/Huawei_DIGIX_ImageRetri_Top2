
import torch
import numpy as np
import pandas as pd
def extract_embedding(model,dataloader):
    model.eval()
    features_list = []*len(dataloader)
    filename_list = []*len(dataloader)
    i=1
    with torch.no_grad():
        print(i-1)
        for image, filename in dataloader:
            image = image.cuda()
            features = model(image, extract_embedding=True)
            features = features.cpu().data.numpy().astype(np.float32)
            features_list.append(features)
            filename_list.append(filename)
        np_filename = np.concatenate(filename_list)
        np_features = np.concatenate(features_list).astype(np.float32)
    df_features = pd.DataFrame(np_features, dtype=np.float32)
    df_filenames = pd.DataFrame(np_filename)


    return pd.concat((df_filenames, df_features), axis=1)