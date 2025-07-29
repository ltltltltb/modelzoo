#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

#安装依赖
git clone https://gitee.com/xiwei777/tcap_dllogger.git
cd tcap_dllogger
python setup.py install
cd ..
cd .. 
pip install -r requirements.txt
cd run_scripts


#执行训练

python run_transformerxl.py --nproc-per-node 1 2>&1 | tee sdaa.log


# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
