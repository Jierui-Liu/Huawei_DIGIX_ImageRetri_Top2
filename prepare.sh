###
 # @Author      : now more
 # @Contact     : lin.honghui@qq.com
 # @LastEditors: Please set LastEditors
 # @Description : 
 # @LastEditTime: 2020-11-18 15:25:09
### 


#0 环境配置
## 本项目所有实验都可单卡训练(1080Ti/V100),CUDA 10.1 CUDNN 7.6.4
## features 依赖安装
conda env create -f env.yaml
conda activate pytorch
## indexing 依赖安装
cd indexing/PyRetri-master
python setup.py install

#1 训练数据生成
cd features/utils
root_dir="./data" #训练集根目录
save_dir="./data" 
patch=576
python features/utils/convert_jpg2npy.py -root_dir $root_dir -save_dir $save_dir -patch $patch
patch=640
python features/utils/convert_jpg2npy.py -root_dir $root_dir -save_dir $save_dir -patch $patch

