# Towards General Purpose Vision Foundation Models for Medical Image Analysis: An Experimental Study of DINOv2 on Radiology Benchmarks

**[Original code is from: https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)**

Mohammed Baharoon, Waseem Qureshi, Jiahong Ouyang, Yanwu Xu, Abdulrhman Aljouie, Wei Peng

In this work, we experiement with the DINOv2 model for radiology, and compare the results to other supervised, self-supervised, weakly-supervised models.

[[`Logs`](https://drive.google.com/drive/folders/1kJpKJIyC-3m3unqm6HmWjhnYGS2jxxwj)][[`Datasets`](https://drive.google.com/drive/folders/1jAeikq-3sSKWV3QSU7gQtrckG3OXKOj7)]

You can find all the logs for all our training experiements, as well as all the model checkpoints, using the `logs` button. You can also find all the train, validation, and test splits, using the `Datasets` button.

# Reproduce Experiments
To test the models for classification, you can use the following command.
```
PYTHONPATH=. python3 dinov2/run/eval/linear.py \
    --gpus <NUM_OF_GPUS> \
    --nodes <NUM_OF_NODES> \
    --batch-size <BATCH_SIZE> \
    --epochs <EPOCHS> \
    --val-epochs <VAL_EPOCHS> \
    --save-checkpoint-frequency <CHECKPOINT_EVERY> \
    --eval-period-epochs <EVAL_PERIOD> \
    --val-metric-type multilabel_auc \
    --finetune False
    --backbone dinov2
    --config-file <PATH_TO_DINOV2_FOLDER>/dinov2/configs/eval/vitb4_pretrain.yaml \
    --pretrained-weights <DINOV2_WEIGHTS_PATH> \
    --output-dir <OUTPUT_PATH> \
    --train-dataset CheXpert:split=TRAIN:root=<PATH_TO_DATASET>/CheXpert \
    --val-dataset CheXpert:split=VAL:root=<PATH_TO_DATASET>/CheXpert \
    --test-dataset CheXpert:split=TEST:root=<PATH_TO_DATASET>/CheXpert 
```
The above command will run a linear evaluation experiment with a DINOv2 ViT-B/14 model on the CheXpert dataset. The run will first search for the optimal hyperparameters by training the model with linear classifiers for `VAL_EPOCHS` number of epochs, testing on the validation set. After that, it will combine the validation and train set and train a new linear for `EPOCHS` number of epochs and evaluate it on the test set.

The parameter `--finetune` determines whether the backbone should be finetuned or not. An additional parameter `--backbone-learning-rate` determines the learning rate for tuning the backbone. The `--backbone` parameter determines the backbone to use, which is set to DINOv2 as default. Other options include `vit-large-imagenet21k`, `resnet-152-imagenet1k`, `vgg-19-imagenet1k`, `densenet-201-imagenet1k`, `msn-large-imagenet1k`, `mae-large-imagenet1k`, `clip-large`, `openclip-huge`, and `sam-large`.

The same command can be applied for segmentation evaluations, simply by changing the path from `dinov2/run/eval/linear.py` to `dinov2/run/eval/segmentation.py`. There are additional segmentation, including `--decoder` (linear or unet) and `--image-size`.

## Citing 

If you use this repository in your work, please consider citing the following.

```
@misc{baharoon2023general,
      title={Towards General Purpose Vision Foundation Models for Medical Image Analysis: An Experimental Study of DINOv2 on Radiology Benchmarks}, 
      author={Mohammed Baharoon and Waseem Qureshi and Jiahong Ouyang and Yanwu Xu and Abdulrhman Aljouie and Wei Peng},
      year={2023},
      eprint={2312.02366},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@misc{oquab2023dinov2,
      title={DINOv2: Learning Robust Visual Features without Supervision}, 
      author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
      year={2023},
      eprint={2304.07193},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
