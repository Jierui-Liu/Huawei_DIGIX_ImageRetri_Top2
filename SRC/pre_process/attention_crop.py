
import sys
sys.path.append("/home/yufei/HUW3")
from SRC.utils.io import *
import numpy as np
from sklearn.cluster import KMeans,MeanShift,estimate_bandwidth,SpectralClustering
import cv2 as cv



    
def nms(bounding_boxes,scores, Nt):
    if len(bounding_boxes) == 0:
        return [], []
    bboxes = np.array(bounding_boxes)
    # 计算 n 个候选框的面积大小
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 对置信度进行排序, 获取排序后的下标序号, argsort 默认从小到大排序
    order = np.argsort(scores)
    picked_boxes = []  # 返回值
    while order.size > 0:
        # 将当前置信度最大的框加入返回值列表中
        index = order[-1]
        picked_boxes.append(bounding_boxes[index])
        # 获取当前置信度最大的候选框与其他任意候选框的相交面积
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h
        # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框删除
        ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ious < Nt)
        order = order[left]
    return picked_boxes

class spectral_cluster(object):
    def __init__(self,n=2,gamma=1.0,intense_thres=0.2,score_thres=0.2):
        self.n=n
        self.gamma=gamma
        self.intense_thres=intense_thres
        self.score_thres=score_thres
    def __call__(self,attention_map):
        boxes=_spectral_cluster(attention_map,self.n,self.gamma,self.intense_thres,self.score_thres)
        ah,aw = attention_map.shape[:2]
        for box in boxes:
            box[1]=box[1] / ah
            box[3]=(box[3] + 1) / ah
            box[0]=box[0] / aw
            box[2]=(box[2] + 1) / aw  #对box归一化
        return boxes


def _spectral_cluster(attention_map,n=2,gamma=1.0,intense_thres=0.2,score_thres=0.2):
    h,w=attention_map.shape
    attention_map=cv.blur(attention_map,(3,3))
    intense_max=attention_map.max()
    intense_min=attention_map.min()
    attention_map=(attention_map-intense_min)/(intense_max-intense_min)
    attention_map[attention_map<=intense_thres]=0
    # attention_map[attention_map>intense_thres]=1
    indexs=np.where(attention_map>intense_thres)
    locations=np.transpose(np.array(indexs))

    labels_ = SpectralClustering(n_clusters=n, gamma=gamma).fit_predict(locations)
    bboxes=[[0,0,h-1,w-1]]
    scores=[1]
    for i in range(np.max(labels_)+1):
        ith_locations=locations[labels_==i,...]
        left=int(ith_locations[:,1].min())
        top=int(ith_locations[:,0].min())
        right=int(ith_locations[:,1].max())
        bottom=int(ith_locations[:,0].max())

        center=ith_locations.mean(axis=0)[np.newaxis]
        # dist_avg=((ith_locations-center)**2).mean()
        map_croped=attention_map[top:bottom,left:right]
        score=map_croped.sum()
        score/=attention_map.sum()
        if score<score_thres:
            continue
        bboxes.append([left-2,top-2,right+2,bottom+2])
        # scores.append(scores)
        scores.append(score)

    bboxes=nms(bboxes,scores,0.45)
    return bboxes


def kmeans_cluster(attention_map,n=2,intense_thres=0.2,score_thres=0.2):
    h,w=attention_map.shape
    intense_max=attention_map.max()
    intense_min=attention_map.min()
    attention_map=(attention_map-intense_min)/(intense_max-intense_min)
    attention_map=cv.blur(attention_map,(3,3))
    intense_thres=0.2
    attention_map[attention_map<=intense_thres]=0
    # attention_map[attention_map>intense_thres]=1
    indexs=np.where(attention_map>intense_thres)
    locations=np.transpose(np.array(indexs))

    clf=KMeans(n_clusters=n)
    clf=clf.fit(locations)
    bboxes=[[0,0,h-1,w-1]]
    scores=[1]
    for i in range(len(clf.cluster_centers_)):
        ith_locations=locations[clf.labels_==i,...]
        left=int(ith_locations[:,1].min())
        top=int(ith_locations[:,0].min())
        right=int(ith_locations[:,1].max())
        bottom=int(ith_locations[:,0].max())

        center=ith_locations.mean(axis=0)[np.newaxis]
        # dist_avg=((ith_locations-center)**2).mean()
        map_croped=attention_map[top:bottom,left:right]
        score=map_croped.sum()
        score/=attention_map.sum()
        if score<score_thres:
            continue
        bboxes.append([left-2,top-2,right+2,bottom+2])
        # scores.append(scores)
        scores.append(score)

    
    bboxes=nms(bboxes,scores,0.45)
    return bboxes




class spectral_cluster_filted(object):
    def __init__(self,n=2,gamma=1.0,intense_thres=0.2,score_thres=0.2,neigh_thres=0.05):
        self.n=n
        self.gamma=gamma
        self.intense_thres=intense_thres
        self.score_thres=score_thres
        self.neigh_thres=neigh_thres
    def __call__(self,attention_map):
        boxes=_spectral_cluster_filted(attention_map,self.n,self.gamma,self.intense_thres,self.score_thres,self.neigh_thres)
        ah,aw = attention_map.shape[:2]
        for box in boxes:
            box[1]=box[1] / ah
            box[3]=(box[3] + 1) / ah
            box[0]=box[0] / aw
            box[2]=(box[2] + 1) / aw  #对box归一化
        return boxes


def _spectral_cluster_filted(attention_map,n=2,gamma=1.0,intense_thres=0.2,score_thres=0.2,neigh_thres=0.05):
    h,w=attention_map.shape
    attention_map=cv.blur(attention_map,(3,3))
    intense_max=attention_map.max()
    intense_min=attention_map.min()
    attention_map=(attention_map-intense_min)/(intense_max-intense_min)
    attention_map[attention_map<=intense_thres]=0
    # attention_map[attention_map>intense_thres]=1
    indexs=np.where(attention_map>intense_thres)
    locations=np.transpose(np.array(indexs))

    labels_ = SpectralClustering(n_clusters=n, gamma=gamma).fit_predict(locations)
    bboxes=[[0,0,h-1,w-1]]
    scores=[1]
    all_locations=[]
    for i in range(np.max(labels_)+1):
        ith_locations=locations[labels_==i,...]
        left=int(ith_locations[:,1].min())
        top=int(ith_locations[:,0].min())
        right=int(ith_locations[:,1].max())
        bottom=int(ith_locations[:,0].max())

        center=ith_locations.mean(axis=0)[np.newaxis]
        # dist_avg=((ith_locations-center)**2).mean()
        map_croped=attention_map[top:bottom,left:right]
        score=map_croped.sum()
        score/=attention_map.sum()
        if score<score_thres:
            continue
        all_locations.append(ith_locations)

        bboxes.append([left-2,top-2,right+2,bottom+2])
        # scores.append(scores)
        scores.append(score)
    if len(all_locations)>=2:
        all_locations_1 = all_locations[0]
        all_locations_2_t = np.transpose(all_locations[1])
        inner_dot = all_locations_1.dot(all_locations_2_t)
        dis = (all_locations_1 ** 2).sum(1)[...,np.newaxis] + (all_locations_2_t ** 2).sum(0)[np.newaxis,...]
        dis = dis - 2 * inner_dot
        dis = np.sqrt(dis)
        # print(np.where(dis<=1.5))
        # print(dis.min(),dis.max())
        if len(np.where(dis<=1.5)[0])>0.05*len(np.where(attention_map>0)[0]):
            return [bboxes[0]]

    bboxes=nms(bboxes,scores,0.45)
    return bboxes




class spectral_cluster_filted_both(object):
    def __init__(self,n=2,gamma=1.0,intense_thres=0.2,score_thres=0.2,neigh_thres=0.05,single_thres=0.8):
        self.n=n
        self.gamma=gamma
        self.intense_thres=intense_thres
        self.score_thres=score_thres
        self.neigh_thres=neigh_thres
        self.single_thres=single_thres
    def __call__(self,attention_map):
        boxes=_spectral_cluster_filted_both(attention_map,self.n,self.gamma,self.intense_thres,self.score_thres,self.neigh_thres,self.single_thres)
        ah,aw = attention_map.shape[:2]
        for box in boxes:
            box[1]=box[1] / ah
            box[3]=(box[3] + 1) / ah
            box[0]=box[0] / aw
            box[2]=(box[2] + 1) / aw  #对box归一化
        return boxes


def _spectral_cluster_filted_both(attention_map,n=2,gamma=1.0,intense_thres=0.2,score_thres=0.2,neigh_thres=0.05,single_thres=0.8):
    h,w=attention_map.shape
    attention_map=cv.blur(attention_map,(3,3))
    intense_max=attention_map.max()
    intense_min=attention_map.min()
    attention_map=(attention_map-intense_min)/(intense_max-intense_min)
    attention_map[attention_map<=intense_thres]=0
    # attention_map[attention_map>intense_thres]=1
    indexs=np.where(attention_map>intense_thres)
    locations=np.transpose(np.array(indexs))

    labels_ = SpectralClustering(n_clusters=n, gamma=gamma).fit_predict(locations)
    bboxes=[[0,0,h-1,w-1]]
    scores=[1]
    all_locations=[]
    for i in range(np.max(labels_)+1):
        ith_locations=locations[labels_==i,...]
        left=int(ith_locations[:,1].min())
        top=int(ith_locations[:,0].min())
        right=int(ith_locations[:,1].max())
        bottom=int(ith_locations[:,0].max())

        center=ith_locations.mean(axis=0)[np.newaxis]
        # dist_avg=((ith_locations-center)**2).mean()
        map_croped=attention_map[top:bottom,left:right]
        score=map_croped.sum()
        score/=attention_map.sum()
        if score<score_thres:
            continue
        all_locations.append(ith_locations)

        bboxes.append([left-2,top-2,right+2,bottom+2])
        # scores.append(scores)
        scores.append(score)
    if len(all_locations)==2:
        all_locations_1 = all_locations[0]
        all_locations_2_t = np.transpose(all_locations[1])
        inner_dot = all_locations_1.dot(all_locations_2_t)
        dis = (all_locations_1 ** 2).sum(1)[...,np.newaxis] + (all_locations_2_t ** 2).sum(0)[np.newaxis,...]
        dis = dis - 2 * inner_dot
        dis = np.sqrt(dis)
        # print(np.where(dis<=1.5))
        # print(dis.min(),dis.max())
        if len(np.where(dis<=1.5)[0])>0.05*len(np.where(attention_map>0)[0]):
            return [bboxes[0]]

    bboxes=nms(bboxes,scores,0.45)
    if len(bboxes)==2:
        left=int(bboxes[1][0])
        top=int(bboxes[1][1])
        right=int(bboxes[1][2])
        bottom=int(bboxes[1][3])
        map_croped=attention_map[top:bottom,left:right]
        score=map_croped.sum()
        score/=attention_map.sum()
        # print(score)
        if score<single_thres:
            return [bboxes[0]]
    return bboxes


class spectral_cluster_filted_both_wdb(object):
    def __init__(self,n=2,gamma=1.0,intense_thres=0.2,score_thres=0.2,neigh_thres=0.05,single_thres=0.8,wdb=2):
        self.n=n
        self.gamma=gamma
        self.intense_thres=intense_thres
        self.score_thres=score_thres
        self.neigh_thres=neigh_thres
        self.single_thres=single_thres
        self.wdb=wdb
    def __call__(self,attention_map):
        boxes=_spectral_cluster_filted_both_wdb(attention_map,self.n,self.gamma,self.intense_thres,self.score_thres,self.neigh_thres,self.single_thres,self.wdb)
        ah,aw = attention_map.shape[:2]
        for box in boxes:
            box[1]=box[1] / ah
            box[3]=(box[3] + 1) / ah
            box[0]=box[0] / aw
            box[2]=(box[2] + 1) / aw  #对box归一化
        return boxes


def _spectral_cluster_filted_both_wdb(attention_map,n=2,gamma=1.0,intense_thres=0.2,score_thres=0.2,neigh_thres=0.05,single_thres=0.8,wdb=2):
    
    h,w=attention_map.shape
    attention_map=cv.blur(attention_map,(3,3))
    intense_max=attention_map.max()
    intense_min=attention_map.min()
    attention_map=(attention_map-intense_min)/(intense_max-intense_min)
    attention_map[attention_map<=intense_thres]=0
    # attention_map[attention_map>intense_thres]=1
    indexs=np.where(attention_map>intense_thres)
    locations=np.transpose(np.array(indexs))

    labels_ = SpectralClustering(n_clusters=n, gamma=gamma).fit_predict(locations)
    bboxes=[[0,0,h-1,w-1]]
    scores=[1]
    all_locations=[]
    dists_avg=[]
    centers=[]
    for i in range(np.max(labels_)+1):
        ith_locations=locations[labels_==i,...]
        left=int(ith_locations[:,1].min())
        top=int(ith_locations[:,0].min())
        right=int(ith_locations[:,1].max())
        bottom=int(ith_locations[:,0].max())

        center=ith_locations.mean(axis=0)[np.newaxis]
        dist_avg=np.sqrt(((ith_locations-center)**2).sum(1)).mean()
        map_croped=attention_map[top:bottom,left:right]
        score=map_croped.sum()
        score/=attention_map.sum()
        if score<score_thres:
            continue
        all_locations.append(ith_locations)
        centers.append(center)
        dists_avg.append(dist_avg)

        bboxes.append([left-2,top-2,right+2,bottom+2])
        # scores.append(scores)
        scores.append(score)
    if len(all_locations)==2:
        all_locations_1 = all_locations[0]
        all_locations_2_t = np.transpose(all_locations[1])
        inner_dot = all_locations_1.dot(all_locations_2_t)
        dis = (all_locations_1 ** 2).sum(1)[...,np.newaxis] + (all_locations_2_t ** 2).sum(0)[np.newaxis,...]
        dis = dis - 2 * inner_dot
        dis = np.sqrt(dis)
        # print(np.where(dis<=1.5))
        # print(dis.min(),dis.max())
        if len(np.where(dis<=1.5)[0])>0.05*len(np.where(attention_map>0)[0]):
            return [bboxes[0]]
        dists_avg=np.array(dists_avg)
        dist_center=np.sqrt(((centers[0]-centers[1])**2).sum())
        # print(dists_avg.mean(),dist_center)
        if dists_avg.mean()>dist_center/wdb:
            return [bboxes[0]]

    bboxes=nms(bboxes,scores,0.45)
    if len(bboxes)==2:
        left=int(bboxes[1][0])
        top=int(bboxes[1][1])
        right=int(bboxes[1][2])
        bottom=int(bboxes[1][3])
        map_croped=attention_map[top:bottom,left:right]
        score=map_croped.sum()
        score/=attention_map.sum()
        # print(score)
        if score<single_thres:
            return [bboxes[0]]
    return bboxes

class spectral_cluster_filted_ns(object):
    def __init__(self,n=2,gamma=1.0,intense_thres=0.2,score_thres=0.2,neigh_thres=0.05,single_thres=0.8,wdb=2):
        self.n=n
        self.gamma=gamma
        self.intense_thres=intense_thres
        self.score_thres=score_thres
        self.neigh_thres=neigh_thres
        self.single_thres=single_thres
        self.wdb=wdb
    def __call__(self,attention_map):
        boxes=_spectral_cluster_filted_ns(attention_map,self.n,self.gamma,self.intense_thres,self.score_thres,self.neigh_thres,self.single_thres,self.wdb)
        ah,aw = attention_map.shape[:2]
        for box in boxes:
            box[1]=box[1] / ah
            box[3]=(box[3] + 1) / ah
            box[0]=box[0] / aw
            box[2]=(box[2] + 1) / aw  #对box归一化
        return boxes


def _spectral_cluster_filted_ns(attention_map,n=2,gamma=1.0,intense_thres=0.2,score_thres=0.2,neigh_thres=0.05,single_thres=0.8,wdb=2):
    
    h,w=attention_map.shape
    attention_map=cv.blur(attention_map,(3,3))
    intense_max=attention_map.max()
    intense_min=attention_map.min()
    attention_map=(attention_map-intense_min)/(intense_max-intense_min)
    attention_map[attention_map<=intense_thres]=0
    # attention_map[attention_map>intense_thres]=1
    indexs=np.where(attention_map>intense_thres)
    locations=np.transpose(np.array(indexs))

    labels_ = SpectralClustering(n_clusters=n, gamma=gamma).fit_predict(locations)
    bboxes=[[0,0,h-1,w-1]]
    scores=[1]
    all_locations=[]
    dists_avg=[]
    centers=[]
    for i in range(np.max(labels_)+1):
        ith_locations=locations[labels_==i,...]
        left=int(ith_locations[:,1].min())
        top=int(ith_locations[:,0].min())
        right=int(ith_locations[:,1].max())
        bottom=int(ith_locations[:,0].max())

        center=ith_locations.mean(axis=0)[np.newaxis]
        dist_avg=np.sqrt(((ith_locations-center)**2).sum(1)).mean()
        map_croped=attention_map[top:bottom,left:right]
        score=map_croped.sum()
        score/=attention_map.sum()
        if score<score_thres:
            continue
        all_locations.append(ith_locations)
        centers.append(center)
        dists_avg.append(dist_avg)

        bboxes.append([left-2,top-2,right+2,bottom+2])
        # scores.append(scores)
        scores.append(score)
    if len(all_locations)==2:
        all_locations_1 = all_locations[0]
        all_locations_2_t = np.transpose(all_locations[1])
        inner_dot = all_locations_1.dot(all_locations_2_t)
        dis = (all_locations_1 ** 2).sum(1)[...,np.newaxis] + (all_locations_2_t ** 2).sum(0)[np.newaxis,...]
        dis = dis - 2 * inner_dot
        dis = np.sqrt(dis)
        # print(np.where(dis<=1.5))
        # print(dis.min(),dis.max())
        if len(np.where(dis<=1.5)[0])>0.05*len(np.where(attention_map>0)[0]):
            return [bboxes[0]]
        dists_avg=np.array(dists_avg)
        dist_center=np.sqrt(((centers[0]-centers[1])**2).sum())
        # print(dists_avg.mean(),dist_center)
        if dists_avg.mean()>dist_center/wdb:
            return [bboxes[0]]

    bboxes=nms(bboxes,scores,0.45)
    if len(bboxes)==2:
        return [bboxes[0]]
    return bboxes



class attention_crop_nobackground(object):
    def __init__(self,intense_thres=0.3):
        self.intense_thres=intense_thres
    def __call__(self,attention_map):
        boxes=_attention_crop(attention_map,self.intense_thres)
        ah,aw = attention_map.shape[:2]
        for box in boxes:
            box[1]=box[1] / ah
            box[3]=(box[3] + 1) / ah
            box[0]=box[0] / aw
            box[2]=(box[2] + 1) / aw  #对box归一化
        return boxes


def _attention_crop(attention_map,intense_thres=0.2,nms_thres=0.75):
    
    h,w=attention_map.shape
    attention_map=cv.blur(attention_map,(3,3))
    intense_max=attention_map.max()
    intense_min=attention_map.min()
    attention_map=(attention_map-intense_min)/(intense_max-intense_min)
    attention_map[attention_map<=intense_thres]=0
    # attention_map[attention_map>intense_thres]=1
    indexs=np.where(attention_map>intense_thres)
    locations=np.transpose(np.array(indexs))

    bboxes=[[0,0,h-1,w-1]]
    scores=[1]

    ith_locations=locations
    left=int(ith_locations[:,1].min())
    top=int(ith_locations[:,0].min())
    right=int(ith_locations[:,1].max())
    bottom=int(ith_locations[:,0].max())

    bboxes.append([left-2,top-2,right+2,bottom+2])
    scores.append(0.9)
    

    bboxes=nms(bboxes,scores,nms_thres)
    return bboxes


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    query_attention_map_file = '/home/yufei/HUW3/data/test_data_A_resize512_rgb/attention_map/query.json'
    attention_dict = file_to_dict(query_attention_map_file)
    attention_maps = attention_dict['attention_map']
    attention_map=attention_maps[1710,0,:,:]

    image=cv.resize(attention_map,(512,512))
    sc=spectral_cluster(n=2)

    boxes=sc(attention_map)

    croped_images=_boxes_crop(image,boxes,attention_map_size=32)



    mat=croped_images[2]
    plt.matshow(mat)

    plt.show()
    b=1

# if __name__ == '__main__':
#     query_attention_map_file = '/home/yufei/HUW3/data/test_data_A_resize512_rgb/attention_map/query.json'
#     attention_dict = file_to_dict(query_attention_map_file)
#     attention_maps = attention_dict['attention_map']
#     fname = attention_dict['fname']
#
#     out_dir = '/home/yufei/HUW3/exp/segmentation/output'
#     if not os.path.exists(out_dir):
#         os.mkdir(out_dir)
#     else:
#         shutil.rmtree(out_dir)
#         os.mkdir(out_dir)
#
#     # ids=list(random.sample(range(0,len(attention_maps)),30))
#     ids=[3127,1710,4603,4266,2356,2352,2720,3485,3521,3630,5529,9586,691]
#     for id in ids:
#         print(id)
#         attention_map=attention_maps[id,0,...]
#         # bboxes=kmeans_cluster(attention_map,n=2)
#         # bboxes=mean_shift_cluster(attention_map)
#         bboxes=spectral_cluster(attention_map)
#         output=np.zeros((attention_map.shape[0],attention_map.shape[1],3))
#         output[...,0]=attention_map
#         output[...,1]=attention_map
#         output[...,2]=attention_map
#         output=(output/output.max()*255).astype(np.uint8)
#         for i in range(len(bboxes)):
#             cv.rectangle(output,(bboxes[i][0],bboxes[i][1]),(bboxes[i][2],bboxes[i][3]),(0,0,255),1)
#         output=cv.resize(output,(200, 200)) #宽,高
#         cv.imwrite(out_dir+'/{}.jpg'.format(id),output)
#

