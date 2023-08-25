#!/bin/bash

# PYTHONPATH=. python3 /mnt/c/Users/user/Desktop/dinov2/dinov2/run/eval/linear.py \
#     --gpus 1 \
#     --nodes 1 \
#     --batch-size 8 \
#     --epochs 10 \
#     --val-metric-type multilabel_auc \
#     --test-metric-types multilabel_auc \
#     --config-file dinov2/configs/eval/vits14_pretrain.yaml \
#     --pretrained-weights models/dinov2_vits14_pretrain.pth \
#     --output-dir results/NIH/dinov2_vits14/knn \
#     --train-dataset NIHChestXray:split=TRAIN:root=/mnt/d/data/NIH/train_tmp \
#     --val-dataset NIHChestXray:split=VAL:root=/mnt/d/data/NIH/test_tmp

PYTHONPATH=. python3 /mnt/c/Users/user/Desktop/dinov2/dinov2/run/eval/segmentation.py \
    --gpus 1 \
    --nodes 1 \
    --batch-size 2 \
    --epochs 1 \
    --val-metric-type segmentation_metrics \
    --test-metric-types segmentation_metrics \
    --config-file dinov2/configs/eval/vits14_pretrain.yaml \
    --pretrained-weights models/dinov2_vits14_pretrain.pth \
    --output-dir results/NIH/dinov2_vits14/knn \
    --train-dataset MC:split=TRAIN:root=/mnt/z/data/MC \
    --val-dataset MC:split=VAL:root=/mnt/z/data/MC

# PYTHONPATH=. python3 /mnt/c/Users/user/Desktop/dinov2/dinov2/run/eval/mlknn.py \
#     --gpus 1 \
#     --nodes 1 \
#     --batch-size 16 \
#     --nb_knn 10 20 100 200 \
#     --config-file dinov2/configs/eval/vits14_pretrain.yaml \
#     --pretrained-weights models/dinov2_vits14_pretrain.pth \
#     --output-dir results/NIH/dinov2_vits14/knn \
#     --train-dataset NIHChestXray:split=TRAIN:root=/mnt/d/data/NIH/train_tmp \
#     --val-dataset NIHChestXray:split=VAL:root=/mnt/d/data/NIH/test_tmp