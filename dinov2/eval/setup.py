# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, List, Optional, Tuple
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.backends.cudnn as cudnn

from dinov2.logging import logging
from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup
import dinov2.utils.utils as dinov2_utils
from dinov2.eval.utils import (ViTLargeImagenet21k, ResNet152ImageNet1k, VGG19ImageNet1k, DenseNet201ImageNet1k, SAMLarge,
                               MAEViTLargeImagenet1k, CLIPLarge, OpenCLIPHuge, ViTLargeMSN, BiomedCLIPBase)
from transformers import ViTForImageClassification


logger = logging.getLogger("dinov2")

def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = [],
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    return parser


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model_for_eval(config, pretrained_weights, backbone):
    if backbone == "vit-large-imagenet21k":
        model = ViTLargeImagenet21k()
        logger.info("Using vit-large-imagenet21k backbone")
    elif backbone == "resnet-152-imagenet1k":
        model = ResNet152ImageNet1k()
        logger.info("Using resnet-152-imagenet1k backbone")
    elif backbone == "vgg-19-imagenet1k":
        model = VGG19ImageNet1k()
        logger.info("Using vgg-19-imagenet1k backbone")
    elif backbone == "densenet-201-imagenet1k":
        model = DenseNet201ImageNet1k()
        logger.info("Using densenet-201-imagenet1k backbone")
    elif backbone == "sam-large":
        model = SAMLarge()
        logger.info("Using sam-large backbone")
    elif backbone == "mae-large-imagenet1k":
        model = MAEViTLargeImagenet1k()
        logger.info("Using mae-large-imagenet1k backbone")
    elif backbone == "clip-large":
        model = CLIPLarge()
        logger.info("Using clip-large backbone")
    elif backbone == "openclip-huge":
        model = OpenCLIPHuge()
        logger.info("Using openclip-huge backbone")
    elif backbone == "msn-large-imagenet1k":
        model = ViTLargeMSN()
        logger.info("Using msn-large-imagenet1k backbone")
    elif backbone == "biomedclip-base":
        model = BiomedCLIPBase()
        logger.info("Using biomedclip-base backbone")
    else:
        model, _ = build_model_from_cfg(config, only_teacher=True)
        dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
        logger.info("Using DINOv2 backbone")
    model.eval()
    model.cuda()
    return model


def setup_and_build_model(args) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup(args)
    model = build_model_for_eval(config, args.pretrained_weights, args.backbone)
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype
