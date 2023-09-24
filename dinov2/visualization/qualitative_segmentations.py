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
from dinov2.data.transforms import make_segmentation_transform, make_segmentation_target_transform
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
        "--decoder",
        type=list,
        help="The type of decoder [linear]",
    )
    parser.set_defaults(
        dataset_str="MC:split=TEST",
        num_of_images = 5,
        random = False,
        learning_rates=[0],
        decoder=["linear"]
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
    decoder,
):
    resize_size = 448
    transform = make_segmentation_transform(resize_size=resize_size)
    target_transform = make_segmentation_target_transform(resize_size=resize_size)

    dataset = make_dataset(
        dataset_str=args.dataset_str,
        transform=transform,
        target_transform=target_transform
    )

    num_of_classes = dataset.get_num_classes()

    embed_dim = model.embed_dim
    decoders, optim_param_groups = setup_decoders(
        embed_dim,
        learning_rates,
        num_of_classes,
    )
    checkpointer = Checkpointer(decoders, head_path)
    checkpointer.resume_or_load(head_path, resume=True)

    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = DINOV2Encoder(model, autocast_ctx=autocast_ctx)

    decoder = list(decoders.module.decoders_dict.values())[0]

    highlight_multipler = 50
    metric = build_segmentation_metrics(average_type=MetricAveraging.SEGMENTATION_METRICS, num_labels=3).cuda()
    for image_index in range(num_of_images):

        image, target = dataset[image_index]
        image, target = image.cuda(non_blocking=True).unsqueeze(0), target.cuda(non_blocking=True).unsqueeze(0)

        with torch.no_grad(): 
            features = feature_model(image)
        logits = decoder(features)
        logits = torch.nn.functional.interpolate(logits, size=resize_size, mode="bilinear", align_corners=False)
        prediction = logits.argmax(dim=1)

        results = metric(prediction, target)

        target = target.squeeze()
        prediction = prediction.squeeze()
        concated = torch.cat((target, prediction), dim=-1)

        prediction = (prediction * highlight_multipler).cpu()
        target = (target * highlight_multipler).cpu()
        concated = (concated.unsqueeze(0).type(torch.int32) * highlight_multipler).cpu()
        H, W = concated.squeeze().shape

        pil_image = torchvision.transforms.ToPILImage()(concated)
        pil_image = pil_image.convert("L") # Convert to Grayscale

        draw = ImageDraw.Draw(pil_image)
        _, _, w, h = draw.textbbox((0, 0), "Target")
        draw.text(((W-w)*0.25, H*0.025), "Target", fill=255)

        _, _, w, h = draw.textbbox((0, 0), "Prediction")
        draw.text(( (W-w) * 0.75, H*0.025), "Prediction", fill=255)

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
        decoder=args.decoder
    )
    return 0

if __name__ == "__main__":
    description = "Segmentation qualitative result"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))