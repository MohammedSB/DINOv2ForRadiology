PYTHONPATH=../ python3 /mnt/c/Users/user/Desktop/dinov2/dinov2/visualization/qualitative_segmentations.py \
    --output-dir /mnt/c/Users/user/Desktop/dinov2/results/NIH/dinov2_vits14/knn \
    --head-path /mnt/c/Users/user/Desktop/dinov2/models/trained_heads/segmentation_linear/model_final.pth\
    --config-file /mnt/c/Users/user/Desktop/dinov2/dinov2/configs/eval/vits14_pretrain.yaml \
    --pretrained-weights /mnt/c/Users/user/Desktop/dinov2/models/dinov2_vits14_pretrain.pth \
    --dataset MC:split=TEST:root=/mnt/z/data/MC/test