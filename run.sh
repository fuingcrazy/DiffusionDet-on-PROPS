#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 训练命令
python train_net.py \
    --config-file configs/diffdet.props.res50.yaml \
    --num-gpus 1 