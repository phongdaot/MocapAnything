#!/bin/bash
# 多数据集(zoo1030 + obj1k [+ mobjaverse])联合训练 —— clean release 版
# 超参与发布训练配方一致。
# 用 mocapanything 环境(cv2 FFMPEG=YES 等)。
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/TripoSG:$PYTHONPATH"

NGPU=${NGPU:-8}
PORT=${PORT:-29510}
CONFIG=${CONFIG:-configs/train/train_video2pose2rot_multidata.yaml}

# setting1 = zoo+obj(config 默认)。setting2 = zoo+obj+mobjaverse:
#   在 config 的 data.train_sets / data.test_sets 各放开 mobjaverse 那条注释即可。

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
torchrun --nproc_per_node="$NGPU" --master_port="$PORT" \
    -m train.video2pose2rot_multidata --config "$CONFIG"
