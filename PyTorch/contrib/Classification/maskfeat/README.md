
# MASKFEAT
## 1. 模型概述
提出了用于视频模型自监督预训练的掩蔽特征预测 (MaskFeat)。首先随机掩蔽输入序列的一部分，然后预测掩蔽区域的特征。无需额外模型权重或监督学习，MaskFeat 在未标注视频上进行预训练，取得了前所未有的优异成绩：在 Kinetics-400 数据集上，MViT-L 准确率为 86.7%，在 Kinetics-600 数据集上，准确率为 88.3%，在 Kinetics-700 数据集上，准确率为 80.4%，在 AVA 数据集上，准确率为 38.8%，在 SSv2 数据集上，准确率为 75.0%。MaskFeat 进一步泛化至图像输入，可将其解读为单帧视频，并在 ImageNet 数据集上取得了颇具竞争力的结果。


- 论文链接：[2112.09133v1\]Masked Feature Prediction for Self-Supervised Visual Pre-Training(https://arxiv.org/abs/2112.09133v1)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/maskfeat

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
Deit 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https://blog.csdn.net/xzxg001/article/details/142465729。


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip3 install  -U openmim 
    pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
    pip3 install opencv_python mmcv --no-deps
    mim install -e .
    pip install -r requirements.txt

    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/maskfeat/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
python run_maskfeat.py --config ../configs/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k.py \
       --launcher pytorch --nproc-per-node 1 --amp \
            --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/imagenet"  2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

![loss](./image/loss.jpg)

MeanRelativeError:-0.11578388039450015
MeanAbsoluteError:-0.049113433254827366
Rule,mean_absolute_error -0.11578388039450015
passmean_relative_error=-0.11578388039450015 <=0.05ormean_absolute_error=-0.049113433254827366<=0.0002

