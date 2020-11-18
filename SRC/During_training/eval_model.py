
import numpy as np
import torch

from SRC.scoring.cos_score import cos_score
from SRC.scoring.eula_score import eula_score

def model_to_train(Model):
    #pass
    Model.module.model_to_train()




def eval_loss_acc(Model, test_data_loader, mul):
    #mul是用于测试的数据的样本数目是minibatchsize的多少倍 20左右
    Model.eval()
    total_acc=0
    total_loss=0
    test_data_loader_iter=iter(test_data_loader)
    with torch.no_grad():
        for i in range(mul):
            im, label = next(test_data_loader_iter)
            im=im.cuda()
            label=label.cuda()
            loss,acc=Model(im,labels=label)
            loss=loss.mean()
            acc=acc.mean()
            total_acc=total_acc+acc.item()
            total_loss=total_loss+loss.item()


    Model.train()
    return total_loss/mul,total_acc/mul


def eval_loss_acc_tri_half(Model, test_data_loader, mul):
    # mul是用于测试的数据的样本数目是minibatchsize的多少倍 20左右
    total_acc = 0
    total_loss1 = 0
    total_loss2 = 0
    with torch.no_grad():
        for i in range(mul):
            im, label = next(iter(test_data_loader))
            im = im.squeeze(0).cuda().half()
            label = label.squeeze(0).cuda().half()
            loss1, loss2, acc = Model(im, labels=label)
            loss1 = loss1.mean()
            loss2 = loss2.mean()
            acc = acc.mean()
            total_acc = total_acc + acc.item()
            total_loss1 = total_loss1 + loss1.item()
            total_loss2 = total_loss2 + loss2.item()

    return total_loss1 / mul, total_loss2 / mul, total_acc / mul

def eval_loss_acc_tri(Model, test_data_loader, mul):
    #mul是用于测试的数据的样本数目是minibatchsize的多少倍 20左右
    Model.eval()
    total_acc=0
    total_loss1=0
    total_loss2=0
    test_data_loader_iter=iter(test_data_loader)
    with torch.no_grad():
        for i in range(mul):
            im, label = next(test_data_loader_iter)
            im=im.squeeze(0).cuda()
            # im=torch.flip(im.clone(),[-1])
            label=label.squeeze(0).cuda()
            loss1,loss2,acc=Model(im,labels=label)
            loss1=loss1.mean()
            loss2=loss2.mean()
            acc=acc.mean()
            total_acc=total_acc+acc.item()
            total_loss1=total_loss1+loss1.item()
            total_loss2=total_loss2+loss2.item()

    return total_loss1/mul,total_loss2/mul,total_acc/mul
            
def eval_loss_acc_tri_multilabels(Model, test_data_loader, mul):
    #mul是用于测试的数据的样本数目是minibatchsize的多少倍 20左右
    total_acc=0
    total_loss1=0
    total_loss2=0
    total_loss3=0
    test_data_loader_iter=iter(test_data_loader)
    with torch.no_grad():
        for i in range(mul):
            im, label = next(test_data_loader_iter)
            im=im.squeeze(0).cuda()
            label=label.squeeze(0).cuda()
            loss1,loss2,loss3,acc=Model(im,labels=label)
            loss1=loss1.mean()
            loss2=loss2.mean()
            loss3=loss3.mean()
            acc=acc.mean()
            total_acc=total_acc+acc.item()
            total_loss1=total_loss1+loss1.item()
            total_loss2=total_loss2+loss2.item()
            total_loss3=total_loss3+loss3.item()
    


    return total_loss1/mul,total_loss2/mul,total_loss3/mul,total_acc/mul

    
def eval_embedding(Model, test_data_loader, mul):
    #mul是用于测试的数据的样本数目是minibatchsize的多少倍 20左右
    with torch.no_grad():
        for i in range(mul):
            im, label = next(iter(test_data_loader))
            im=im.squeeze(0).cuda()
            label=label.squeeze(0).cuda()
            embedding=Model(im, extract_embedding=True)
            if i==0:
                embeddings=embedding.detach()
                labels=label.cpu().numpy().tolist()
            else:
                embeddings=torch.cat((embeddings,embedding.detach()),dim=0)
                labels=labels+label.cpu().numpy().tolist()
    


    return embeddings,labels



def eval_top1_acc_mAP10(Model,query_dataloader,gallery_dataloader):
    #这个函数提取query_dataloader和gallery_dataloader中的所有embedding，
    #并且按照固定的方式划分为query和gallery，并且通过cos距离计算top1准确率
    #适用于
    #效率
    #提取embedding
    #计算cos

    Model.eval()
    with torch.no_grad():
        query_embeddings = []
        query_labels = []
        for im, label in query_dataloader:
            im = im.cuda()
            embedding=Model(im,extract_embedding=True)
            query_embeddings.append(embedding.cpu().data.numpy())
            query_labels.append(label.cpu())
        query_embedding = np.concatenate(query_embeddings)
        query_label=np.concatenate(query_labels)

        gallery_embeddings = []
        gallery_labels = []
        for im, label in gallery_dataloader:
            im = im.cuda()
            embedding=Model(im,extract_embedding=True)
            gallery_embeddings.append(embedding.cpu().data.numpy())
            gallery_labels.append(label)
        gallery_embedding = np.concatenate(gallery_embeddings)
        gallery_label=np.concatenate(gallery_labels)

    score=cos_score(query_embedding,gallery_embedding)

    #接下来计算top1_acc 和 top10_map
    
    #sorted_index = np.argsort(1-score, dim=1)
    sorted_index = np.argsort(1-score, axis=1)
    k=10
    sorted_index_top1 = sorted_index[:, 0]
    sorted_index_topk = sorted_index[:, :k]

    # 求top1_acc
    top1_label=gallery_label[sorted_index_top1]
    # a=np.hstack((top1_label[:,np.newaxis],query_label[:,np.newaxis]))
    # print(a)
    correct = (top1_label == query_label).sum()
    top1_acc = 1.0*correct / query_label.shape[0]   

    # 求top10_map
    nums_every_class=[min(len(np.where(gallery_label==label)[0]),k) for label in query_label]
    topk_map_every_class=[]
    tmp_0=np.array(range(k))+1
    for n in range(len(query_label)):
        if nums_every_class[n]==0:
            print('nums0_every_class:',n)
            continue
        topk_label=gallery_label[sorted_index_topk[n,:]]
        indexs=np.where(topk_label==query_label[n])[0]+1
        if len(indexs)==0:
            topk_map_every_class.append(0)
            continue
        tmp_1=tmp_0[:len(indexs)]
        topk_map_here=(1.0*tmp_1/indexs).sum()/nums_every_class[n]
        topk_map_every_class.append(topk_map_here)
    if len(topk_map_every_class)==0:
        topk_map=0
    else:
        topk_map_every_class=np.array(topk_map_every_class)
        topk_map=topk_map_every_class.mean()


    # top1_acc, mAP10=0,0
    model_to_train(Model)
    return top1_acc,topk_map


def eval_top1_acc_mAP10_eula(Model, query_dataloader, gallery_dataloader):
    # 这个函数提取query_dataloader和gallery_dataloader中的所有embedding，
    # 并且按照固定的方式划分为query和gallery，并且通过cos距离计算top1准确率
    # 适用于
    # 效率
    # 提取embedding
    # 计算cos

    Model.eval()
    with torch.no_grad():
        query_embeddings = []
        query_labels = []
        for im, label in query_dataloader:
            im = im.cuda()
            embedding = Model(im, extract_embedding=True)
            query_embeddings.append(embedding.cpu().data.numpy())
            query_labels.append(label.cpu())
        query_embedding = np.concatenate(query_embeddings)
        query_label = np.concatenate(query_labels)

        gallery_embeddings = []
        gallery_labels = []
        for im, label in gallery_dataloader:
            im = im.cuda()
            embedding = Model(im, extract_embedding=True)
            gallery_embeddings.append(embedding.cpu().data.numpy())
            gallery_labels.append(label)
        gallery_embedding = np.concatenate(gallery_embeddings)
        gallery_label = np.concatenate(gallery_labels)

    score = eula_score(query_embedding, gallery_embedding)

    # 接下来计算top1_acc 和 top10_map

    # sorted_index = np.argsort(1-score, dim=1)
    sorted_index = np.argsort(1 - score, axis=1)
    k = 10
    sorted_index_top1 = sorted_index[:, 0]
    sorted_index_topk = sorted_index[:, :k]

    # 求top1_acc
    top1_label = gallery_label[sorted_index_top1]
    # a=np.hstack((top1_label[:,np.newaxis],query_label[:,np.newaxis]))
    # print(a)
    correct = (top1_label == query_label).sum()
    top1_acc = 1.0 * correct / query_label.shape[0]

    # 求top10_map
    nums_every_class = [min(len(np.where(gallery_label == label)[0]), k) for label in query_label]
    topk_map_every_class = []
    tmp_0 = np.array(range(k)) + 1
    for n in range(len(query_label)):
        if nums_every_class[n] == 0:
            print('nums0_every_class:', n)
            continue
        topk_label = gallery_label[sorted_index_topk[n, :]]
        indexs = np.where(topk_label == query_label[n])[0] + 1
        if len(indexs) == 0:
            topk_map_every_class.append(0)
            continue
        tmp_1 = tmp_0[:len(indexs)]
        topk_map_here = (1.0 * tmp_1 / indexs).sum() / nums_every_class[n]
        topk_map_every_class.append(topk_map_here)
    if len(topk_map_every_class) == 0:
        topk_map = 0
    else:
        topk_map_every_class = np.array(topk_map_every_class)
        topk_map = topk_map_every_class.mean()

    # top1_acc, mAP10=0,0
    model_to_train(Model)
    return top1_acc, topk_map







if __name__ == '__main__':
    gallery_label=np.array([0,1,2,0,0,2,3,0,1,2])
    # query_label=np.array([2,1,0])
    query_label=np.array([0,1,2])
    # sorted_index_top1=np.array([5,1,1])
    sorted_index_top1=np.array([3])
    sorted_index_topk=np.array([[0,3,7],[1,8,0],[5,1,9]])
    
    # 求top1_acc
    top1_label=gallery_label[sorted_index_top1]
    correct = (top1_label == query_label).sum()
    top1_acc = 1.0*correct / query_label.shape[0] 
    print('top1_acc:',top1_acc)

    # 求top10_map
    k=3
    nums_every_class=[min(len(np.where(gallery_label==label)[0]),k) for label in range(len(query_label))]
    print('top1_acc:',nums_every_class)
    topk_map_every_class=[]
    tmp_0=np.array(range(k))+1
    for n in range(len(query_label)):
        topk_label=gallery_label[sorted_index_topk[n,:]]
        indexs=np.where(topk_label==query_label[n])[0]+1
        tmp_1=tmp_0[:len(indexs)]
        topk_map_here=(1.0*tmp_1/indexs).sum()/nums_every_class[n]
        topk_map_every_class.append(topk_map_here)
    topk_map_every_class=np.array(topk_map_every_class)
    topk_map=topk_map_every_class.mean()
    print('topk_map__:',topk_map_every_class)
    print('topk_map:',topk_map)
    
    
