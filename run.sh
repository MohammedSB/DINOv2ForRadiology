#!/bin/bash

PYTHONPATH=. python3 /mnt/c/Users/user/Desktop/dinov2/dinov2/run/eval/linear.py \
    --gpus 1 \
    --nodes 1 \
    --batch-size 8 \
    --config-file dinov2/configs/eval/vits14_pretrain.yaml \
    --pretrained-weights models/dinov2_vits14_pretrain.pth \
    --output-dir results/NIH/dinov2_vits14/knn \
    --train-dataset NIHChestXray:split=TRAIN:root=/mnt/d/data/NIH/train_tmp \
    --val-dataset NIHChestXray:split=VAL:root=/mnt/d/data/NIH/test_tmp

