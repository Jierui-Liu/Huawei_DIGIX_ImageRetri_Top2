# Huawei_Digix_ImgRetri_Top2
2020 DIGIX GLOBAL AI CHALLENGE - Digital Device Image Retrieval - WEARE队 亚军



# 0. 环境&依赖

**环境：**

+ Ubuntu 16.04
+ CUDA 10.1 CUDNN 7.6.4
+ 1080Ti or V100
+ 内存128G

本项目所有实验均为单卡运行，**使用1080Ti时batchsize改为 8（类）x 4（张）**。

**依赖：**

```
# 训练环境
conda env create -f env.yaml
conda activate pytorch

# 后处理
cd indexing/PyRetri-master
python setup.py install
```



# 1. 数据预处理

```
# 训练数据生成
cd features/utils

root_dir="./data" # 数据集所在目录
save_dir="./data" 
patch=640

python features/utils/convert_jpg2npy.py -root_dir $root_dir -save_dir $save_dir -patch $patch
```





## 2. 文件目录
本项目部分代码参考开源仓库 [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline)、[fast-reid](https://github.com/JDAI-CV/fast-reid)、[PyRetri](https://github.com/PyRetri/PyRetri)

```
.
├── prepare.sh (依赖包安装、数据预处理)
├── features
│   ├── checkpoints (保存训练过程权重)
│   ├── configs     (配置文件)
│   ├── data
│   ├── engine
│   ├── exp         (特征保存目录)
│   ├── log
│   ├── model
│   ├── README.md
│   ├── solver
│   ├── tools
│   └── utils
├── indexing
│   ├── features
│   ├── index_configs (特征检索配置文件)
│   ├── index_tools
│   ├── PyRetri-master
│   ├── README.md
│   ├── result_tmp    (检索结果保存目录)
│   └── trans_index_json.sh (特征检索启动文件)
└── env.yaml
```



# 3. 算法说明
[Coming soon](https://www.zhihu.com/people/lin-hong-hui-81/posts)

