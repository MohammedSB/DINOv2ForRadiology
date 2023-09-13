#!/bin/bash

# PYTHONPATH=. python3 /mnt/c/Users/user/Desktop/dinov2/dinov2/run/eval/linear.py \
#     --gpus 1 \
#     --nodes 1 \
#     --batch-size 4 \
#     --val-epochs 50 \
#     --eval-period-epochs 100 \
#     --epochs 4 \
#     --n-last-blocks 1 \
#     --avgpools False \
#     --val-metric-type multilabel_auc \
#     --config-file dinov2/configs/eval/vits14_pretrain.yaml \
#     --pretrained-weights models/dinov2_vits14_pretrain.pth \
#     --output-dir results/NIH/dinov2_vits14/knn \
#     --train-dataset NIHChestXray:split=TRAIN:root=/mnt/d/data/NIH \
#     --val-dataset NIHChestXray:split=VAL:root=/mnt/d/data/NIH \
#     --test-dataset NIHChestXray:split=TEST:root=/mnt/d/data/NIH \

PYTHONPATH=. python3 /mnt/c/Users/user/Desktop/dinov2/dinov2/run/eval/linear.py \
    --gpus 1 \
    --nodes 1 \
    --batch-size 4 \
    --val-epochs 50 \
    --eval-period-epochs 100 \
    --epochs 4 \
    --n-last-blocks 1 \
    --avgpools False \
    --val-metric-type multilabel_auc \
    --config-file dinov2/configs/eval/vits14_pretrain.yaml \
    --pretrained-weights models/dinov2_vits14_pretrain.pth \
    --output-dir results/NIH/dinov2_vits14/knn \
    --train-dataset SARSCoV2CT:split=TRAIN:root=/mnt/z/data/SARS-CoV-2-CT \
    --val-dataset SARSCoV2CT:split=VAL:root=/mnt/z/data/SARS-CoV-2-CT \
    --test-dataset SARSCoV2CT:split=TEST:root=/mnt/z/data/SARS-CoV-2-CT \

# PYTHONPATH=. python3 /mnt/c/Users/user/Desktop/dinov2/dinov2/run/eval/segmentation.py \
#     --gpus 1 \
#     --nodes 1 \
#     --batch-size 1 \
#     --epochs 1 \
#     --val-metric-type segmentation_metrics \
#     --test-metric-types segmentation_metrics \
#     --config-file dinov2/configs/eval/vits14_pretrain.yaml \
#     --pretrained-weights models/dinov2_vits14_pretrain.pth \
#     --output-dir results/NIH/dinov2_vits14/knn \
#     --train-dataset Shenzhen:split=TRAIN:root=/mnt/z/data/Shenzhen \
#     --val-dataset Shenzhen:split=VAL:root=/mnt/z/data/Shenzhen \
#     --test-dataset Shenzhen:split=TEST:root=/mnt/z/data/Shenzhen

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