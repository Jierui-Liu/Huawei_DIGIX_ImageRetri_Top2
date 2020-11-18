'''
@Author      : now more
@Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
@Description : 
LastEditTime: 2020-08-29 10:56:29
'''


config = dict(
    # Basic Config
    enable_backends_cudnn_benchmark = True,
    max_epochs = 300+1,
    log_period = 0.05,
    save_dir = r"../checkpoints/",
    log_dir = r"../log/",

    # Dataset
    # train_dataloader：
    #           --> dataloader: image2batch,继承自 torch.utils.data.DataLoader,
    #               --> batch_size: 每个batch数目
    #               --> shuffle : 是否打乱数据，默认训练集打乱，测试集不打乱
    #               --> num_workers : 多线程加载数据
    #               --> drop_last : 若 len_epoch 无法整除 batch_size 时，丢弃最后一个batch。（在较早的torch版本中，开启多卡GPU加速后，
    #                               若batch无法整除多卡数目，代码运行会报错，避免出错风险丢弃最后一个batch）
    #
    #           --> transforms: 在线数据增强加载，传入数据增强函数及对应参数配置，相关代码在 data/transforms/opencv_transforms.py
    #
    #           --> dataset: 加载image和label，继承自 torch.utils.data.Dataset,对应代码在 data/dataset/bulid.py
    #                           
    train_pipeline = dict(
        dataloader = dict(batch_size = 10,num_workers = 15,drop_last = True,pin_memory=False,
                        collate_fn="my_collate_fn"),

        dataset = dict(type="train_dataset",
                    root_dir = r"/home/LinHonghui/Datasets/HW_ImageRetrieval/train_data_resize448",
                    images_per_classes=4,classes_per_minibatch=1),

        transforms = [
            # dict(type="RescalePad",output_size=320),
            # dict(type="MultisizePad",p=0.1,resizes=[384,448,320],padsize=512),
            dict(type="ShiftScaleRotate",p=0.3,shift_limit=0.1,scale_limit=(-0.5,0.2),rotate_limit=15),
            dict(type="IAAPerspective",p=0.1,scale=(0.05, 0.15)),
            dict(type="ChannelShuffle",p=0.1),
            # dict(type="RandomCropResized",p=0.15,output_size=(512,512),scale=(0.85,0.95),ratio=(3/4,4/3)),
            dict(type="RandomRotate90",p=0.2),
            # dict(type="RandomRotation",degrees=(-15,15)),
            dict(type="RandomHorizontalFlip",p=0.5),
            dict(type="RandomVerticalFlip",p=0.5),
            dict(type="ColorJitter",brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
            dict(type="RandomErasing",p=0.2,sl=0.02,sh=0.2,rl=0.2),
            dict(type="RandomPatch",p=0.05,pool_capacity=1000,min_sample_size=100,patch_min_area=0.01,patch_max_area=0.2,patch_min_ratio=0.2,p_rotate=0.5,p_flip_left_right=0.5),
            dict(type="ToTensor",),
            dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=True),
            ],

    ),

    gallery_pipeline = dict(
        dataloader = dict(batch_size = 288,shuffle = False,num_workers = 16,drop_last = False),
        dataset = dict(type="load_npy",
                    image_dir = r"/home/LinHonghui/Datasets/HW_ImageRetrieval/test_data_A_resize448/gallery",),
        transforms = [
            dict(type="ToTensor",),
            dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=False),
            ],

    ),
    query_pipeline = dict(
        dataloader = dict(batch_size = 288,shuffle = False,num_workers = 16,drop_last = False),
        dataset = dict(type="load_npy",
                    image_dir = r"/home/LinHonghui/Datasets/HW_ImageRetrieval/test_data_A_resize448/query",),
        transforms = [
            # dict(type="RescalePad",output_size=320),
            dict(type="ToTensor",),
            dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=False),
            ],
    ),
    # Model
    # model : 
    ##      --> backbone : 特征提取，需在model/backbone中定义
    ##      --> aggregation : pooling_layer,model/aggregation
    ##      --> heads : classification heads,model/heads
    ##      --> losses: criterion. model/losses 
    model = dict(
        net = dict(type="CLUNet"),
        backbone = dict(type="densenet169",pretrained=True),
        aggregation = dict(type="GeneralizedMeanPoolingP",output_size=(1,1),),
        losses = dict(type="Norm_Softmax",in_features=1664,num_clssses=3097),
    ),


    multi_gpu = True,
    max_num_devices = 1, #自动获取空闲显卡，默认第一个为主卡


    # Solver
    ## lr_scheduler : 学习率调整策略，默认从 torch.optim.lr_scheduler 中加载
    ## optimizer : 优化器，默认从 torch.optim 中加载
    # lr_scheduler = dict(type="ExponentialLR",gamma=0.9999503585), # cycle_momentum=False if optimizer==Adam
    lr_scheduler = dict(type="ExponentialLR",gamma=0.99995), # cycle_momentum=False if optimizer==Adam

    optimizer = dict(type="Adam",lr=2e-4),
    # optimizer = dict(type="AdamW",lr=2e-4),
    warm_up = dict(length=1,min_lr=4e-6,max_lr=2e-4,froze_num_lyers=8)
    # Outpu
    # save_dir = r""
    

)


if __name__ == "__main__":
    pass