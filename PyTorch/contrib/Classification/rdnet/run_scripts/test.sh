#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo $script_path

#安装依赖
pip3 install -r ../requirements.txt
# 数据集路径,保持默认统一根目录即可
data_path="/data/teco-data/imagenet"
# # 参数校验
# for para in $*
# do
#     if [[ $para == --data_path* ]];then
#         data_path=`echo ${para#*=}`
# done

#如长训请提供完整命令即可，100iter对齐提供100iter命令即可

#示例1: python run_resnet.py --nproc_per_node 4 --model_name resnet50 --epoch 1 --batch_size 32 --device sdaa --step 100 --datasets $dataset 2>&1 | tee sdaa.log
#由于demo无需下载数据集及数据集太小所以未做step适配，正常场景参考示例1即可
python3 run_rdnet.py --data_path /data/teco-data/imagenet --batch_size 16 --epochs 1 --lr 0.1 --save_path ./checkpoints --num_steps 100 2>&1 | tee sdaa.log

#生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log