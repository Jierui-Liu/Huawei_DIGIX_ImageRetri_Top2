
import torch
import numpy as np
import pandas as pd
def extract_embedding(model,dataloader):
    model.eval()
    features_list = []*len(dataloader)
    filename_list = []*len(dataloader)
    with torch.no_grad():
        for image, filename in dataloader:
            image = image.cuda()

            features_1 = model(image.clone(), extract_embedding=True)
            features_1 = features_1.cpu().data.numpy().astype(np.float32)

            # 水平
            features_2 = model(torch.flip(image.clone(),[-1]), extract_embedding=True)
            features_2 = features_2.cpu().data.numpy().astype(np.float32)

            # 垂直
            features_3 = model(torch.flip(image.clone(),[-2]), extract_embedding=True)
            features_3 = features_3.cpu().data.numpy().astype(np.float32)
            
            # 顺时针90
            features_4 = model(torch.flip(image.clone().transpose(2,3),[-1]), extract_embedding=True)
            features_4 = features_4.cpu().data.numpy().astype(np.float32)

            # 逆时针90
            features_5 = model(torch.flip(image.clone().transpose(2,3),[-2]), extract_embedding=True)
            features_5 = features_5.cpu().data.numpy().astype(np.float32)

            features = np.concatenate((features_1,features_2,features_3,features_4,features_5),axis=1)
            # import pdb; pdb.set_trace()
            features_list.append(features)
            filename_list.append(filename)



            # image = image.cuda()
            # features = model(image, extract_embedding=True)
            # features = features.cpu().data.numpy().astype(np.float32)
            # features_list.append(features)
            # filename_list.append(filename)
        np_filename = np.concatenate(filename_list)
        np_features = np.concatenate(features_list).astype(np.float32)
    df_features = pd.DataFrame(np_features, dtype=np.float32)
    df_filenames = pd.DataFrame(np_filename)


    return pd.concat((df_filenames, df_features), axis=1)