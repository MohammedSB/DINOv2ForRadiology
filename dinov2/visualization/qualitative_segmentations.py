import toolz
import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional
import math
from PIL import Image, ImageFont, ImageDraw

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data.transforms import make_segmentation_train_transforms, make_segmentation_eval_transforms
from dinov2.eval.metrics import MetricAveraging, build_metric, build_segmentation_metrics
from dinov2.eval.setup import setup_and_build_model, get_args_parser as get_setup_args_parser
from dinov2.logging import MetricLogger
from dinov2.eval.segmentation.utils import setup_decoders, DINOV2Encoder, LinearDecoder

def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = [],
    add_help: bool = True,
):
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--head-path",
        dest="head_path",
        type=str,
        help="Path of the segmentation decoder or head",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_str",
        type=str,
        help="Generate qualitative results on dataset",
    )
    parser.add_argument(
        "--num-of-images",
        dest="num_of_images",
        type=int,
        help="Number of images to generate qualtitave",
    )
    parser.add_argument(
        "--random",
        dest="random",
        type=bool,
        help="Whether to select random images from the dataset",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search",
    )
    parser.add_argument(
        "--decoder-type",
        type=str,
        help="The type of decoder [linear, unet]",
    )
    parser.set_defaults(
        dataset_str="MC:split=TEST",
        num_of_images = 5,
        random = False,
        learning_rates=[0],
        decoder_type="linear"
    )
    return parser

def run_qualtitave_result_generation(
    model,
    head_path,
    autocast_dtype,
    output_dir,
    dataset,
    num_of_images,
    random,
    learning_rates,
    decoder_type,
):
    seed = 0
    torch.manual_seed(seed)
    
    resize_size = 448
    train_image_transform, train_target_transform = make_segmentation_train_transforms()
    eval_image_transform, eval_target_transform  = make_segmentation_eval_transforms()

    dataset = make_dataset(
        dataset_str=args.dataset_str,
        transform=eval_image_transform,
        target_transform=eval_target_transform
    )

    num_of_classes = dataset.get_num_classes()
    is_3d = dataset.is_3d()

    embed_dim = model.embed_dim
    decoders, optim_param_groups = setup_decoders(
        embed_dim,
        learning_rates,
        num_of_classes,
        decoder_type,
        is_3d=is_3d,
        image_size=resize_size
    )
    checkpointer = Checkpointer(decoders, head_path)
    checkpointer.resume_or_load(head_path, resume=True)

    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    n_last_blocks = 5 if decoder_type == "unet" else 1
    feature_model = DINOV2Encoder(model, autocast_ctx=autocast_ctx, n_last_blocks=n_last_blocks)

    decoder = list(decoders.module.decoders_dict.values())[0]

    highlight_multipler = 255//num_of_classes
    metric = build_segmentation_metrics(average_type=MetricAveraging.SEGMENTATION_METRICS, num_labels=num_of_classes).cuda()
    for image_index in range(num_of_images):

        image, target = dataset[image_index]
        image, target = image.cuda(non_blocking=True).unsqueeze(0), target.cuda(non_blocking=True).unsqueeze(0)
        
        with torch.no_grad(): 
            features = feature_model(image)
        logits = decoder(features)
        logits = torch.nn.functional.interpolate(logits, size=resize_size, mode="bilinear", align_corners=False)
        prediction = logits.argmax(dim=1)

        results = metric(prediction, target)

        prediction = prediction.squeeze()
        prediction = (prediction * highlight_multipler).cpu()
        H, W = prediction.squeeze().shape
        pil_image = torchvision.transforms.ToPILImage()(prediction.type(torch.int32))
        pil_image = pil_image.convert("L") # Convert to Grayscale
        
        draw = ImageDraw.Draw(pil_image)

        result_meta = ""
        for m, r in dict(results).items():
            result_meta += f"{m}: {float(r):.3f} "

        draw.text((0, 0), result_meta, fill=255)

        pil_image.save(f"{output_dir}/{dataset.images[image_index]}")


def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    run_qualtitave_result_generation(
        model=model,
        head_path=args.head_path,
        autocast_dtype=autocast_dtype,
        output_dir=args.output_dir,
        dataset=args.dataset_str,
        num_of_images=args.num_of_images,
        random=args.random,
        learning_rates=args.learning_rates,
        decoder_type=args.decoder_type
    )
    return 0

if __name__ == "__main__":
    description = "Segmentation qualitative result"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))