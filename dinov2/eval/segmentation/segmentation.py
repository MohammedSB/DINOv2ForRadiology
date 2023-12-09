# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import toolz
import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional
import math

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from monai.losses.dice import DiceLoss, DiceCELoss

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data.transforms import (make_classification_eval_transform, make_classification_train_transform,
                                    make_segmentation_train_transforms, make_segmentation_eval_transforms)
import dinov2.distributed as distributed
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import (extract_hyperparameters_from_model, ModelWithIntermediateLayers, evaluate,
                                apply_method_to_nested_values, make_datasets, make_data_loaders, collate_fn_3d)
from dinov2.eval.segmentation.utils import (setup_decoders, LinearPostprocessor, DINOV2Encoder, save_test_results)
from dinov2.logging import MetricLogger
from dinov2.data.wrappers import FewShotDatasetWrapper


logger = logging.getLogger("dinov2")


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
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    parser.add_argument(
        "--test-dataset",
        dest="test_dataset_str",
        type=str,
        help="Test datasets, none to reuse the validation dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--val-epochs",
        type=int,
        help="Number of epochs for testing on validation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=int,
        help="Number of epochs between two named checkpoint saves.",
    )
    parser.add_argument(
        "--eval-period-epochs",
        type=int,
        help="Number of iterations between two evaluations.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume from existing checkpoints",
    )
    parser.add_argument(
        "--val-metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Validation metric",
    )
    parser.add_argument(
        "--test-metric-types",
        type=MetricType,
        choices=list(MetricType),
        nargs="+",
        help="Evaluation metric",
    )
    parser.add_argument(
        "--classifier-fpath",
        type=str,
        help="Path to a file containing pretrained linear classifiers",
    )
    parser.add_argument(
        "--val-class-mapping-fpath",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        "--test-class-mapping-fpaths",
        nargs="+",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        "--decoder-type",
        type=str,
        help="The type of decoder to use [linear, unet]",
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        help="Number of shots for each class.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="Size of input image",
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        help="The loss function used",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        help="The name of the backbone model to use [dinov2, vit-large-imagenet21k]",
    )
    parser.set_defaults(
        train_dataset_str="MC:split=TRAIN",
        test_dataset_str="MC:split=TEST",
        val_dataset_str=None,
        epochs=10,
        val_epochs=None,
        batch_size=128,
        num_workers=0,
        epoch_length=None,
        save_checkpoint_frequency=5,
        eval_period_epochs=5,
        learning_rates=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        val_metric_type=MetricType.MULTILABEL_AUROC,
        segmentor_fpath=None,
        decoder_type="linear",
        shots=None,
        image_size=448,
        loss_function="dice",
        backbone="dinov2"
    )
    return parser


def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m

@torch.no_grad()
def evaluate_segmentors(
    feature_model,
    decoders,
    data_loader,
    metric_type,
    metrics_file_path,
    num_of_classes,
    iteration,
    prefixstring="",
    best_segmentor_on_val=None,
):
    logger.info("running validation !")

    labels = list(data_loader.dataset.class_names)
    metric = build_metric(metric_type, num_classes=num_of_classes, labels=labels)
    postprocessors = {k: LinearPostprocessor(v) for k, v in decoders.decoders_dict.items()}
    metrics = {k: metric.clone() for k in decoders.decoders_dict}

    _, results_dict_temp = evaluate(
        feature_model,
        data_loader,
        postprocessors,
        metrics,
        torch.cuda.current_device(),
    )

    logger.info("")
    results_dict = {}
    max_score = 0
    best_segmentor = ""
    eval_metric = str(list(metric)[0])

    for i, (segmentor_string, metric) in enumerate(results_dict_temp.items()):
        logger.info(f"{prefixstring} -- Segmentor: {segmentor_string} * {metric}")
        if (
            best_segmentor_on_val is None and metric[eval_metric].item() > max_score
        ) or segmentor_string == best_segmentor_on_val:
            max_score = metric[eval_metric].item()
            best_segmentor = segmentor_string

    results_dict["best_segmentor"] = {"name": best_segmentor, "results": apply_method_to_nested_values(
                                                                            results_dict_temp[best_segmentor],
                                                                            method_name="item",
                                                                            nested_types=(dict))}

    logger.info(f"best segmentor: {results_dict['best_segmentor']}") 

    if distributed.is_main_process():
        with open(metrics_file_path, "a") as f:
            f.write(f"iter: {iteration}\n")
            for k, v in results_dict.items():
                f.write(json.dumps({k: v}) + "\n")
            f.write("\n")

    return results_dict

def eval_decoders(
    *,
    feature_model,
    decoders,
    train_data_loader,
    val_data_loader,
    metrics_file_path,
    optimizer,
    scheduler,
    output_dir,
    max_iter,
    checkpoint_period,  # In number of iter, creates a new file every period
    running_checkpoint_period,  # Period to update main checkpoint file
    eval_period,
    metric_type,
    num_of_classes,
    resume=True,
    segmentor_fpath=None,
    is_3d=False,
    loss_function=DiceLoss()
):
    checkpointer = Checkpointer(decoders, output_dir, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(segmentor_fpath or "", resume=resume).get("iteration", 0) + 1

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, checkpoint_period, max_iter=max_iter)
    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(delimiter="  ")
    header = "Training"

    for data, labels in metric_logger.log_every(
        train_data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        data = data.cuda(non_blocking=True)

        features = feature_model(data)
        outputs = decoders(features)

        if is_3d:
            outputs = {m: torch.cat(output, dim=0) for m, output in outputs.items()}
            labels = torch.cat(labels, dim=0)

        labels = labels.cuda(non_blocking=True).type(torch.int64)
        losses = {f"loss_{k}": loss_function(v, labels.unsqueeze(1)) for k, v in outputs.items()}
        
        loss = sum(losses.values())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()
        scheduler.step()

        # log
        if iteration % 10 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            print("lr", optimizer.param_groups[0]["lr"])

        if iteration - start_iter > 5:
            if iteration % running_checkpoint_period == 0:
                torch.cuda.synchronize()
                if distributed.is_main_process():
                    logger.info("Checkpointing running_checkpoint")
                    periodic_checkpointer.save("running_checkpoint_linear_eval", iteration=iteration)
                torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        if eval_period > 0 and iteration % eval_period == 0 and iteration != max_iter:
            _ = evaluate_segmentors(
                feature_model=feature_model,
                decoders=remove_ddp_wrapper(decoders),
                data_loader=val_data_loader,
                metrics_file_path=metrics_file_path,
                prefixstring=f"ITER: {iteration} {val_data_loader.dataset.split.value}",
                metric_type=metric_type,
                num_of_classes=num_of_classes,
                iteration=iteration,
            )
            torch.cuda.synchronize()

        iteration = iteration + 1

    val_results_dict = evaluate_segmentors(
        feature_model=feature_model,
        decoders=remove_ddp_wrapper(decoders),
        data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        prefixstring=f"ITER: {iteration} {val_data_loader.dataset.split.value}",
        metric_type=metric_type,
        num_of_classes=num_of_classes,
        iteration=iteration,
    )
    return val_results_dict, feature_model, decoders, iteration


def run_eval_segmentation(
    model,
    decoder_type,
    output_dir,
    train_dataset_str,
    test_dataset_str,
    batch_size,
    epochs,
    val_epochs,
    epoch_length,
    num_workers,
    save_checkpoint_frequency,
    eval_period_epochs,
    learning_rates,
    autocast_dtype,
    val_dataset_str=None,
    resume=True,
    segmentor_fpath=None,
    val_metric_type=MetricType.SEGMENTATION_METRICS,
    shots=None,
    image_size=448,
    loss_function="dice",
    backbone="dinov2"
):
    seed = 0
    torch.manual_seed(seed)

    if test_dataset_str == None:
        raise ValueError("Test dataset cannot be None")
    
    # make datasets
    train_image_transform, train_target_transform = make_segmentation_train_transforms(resize_size=image_size)
    eval_image_transform, eval_target_transform  = make_segmentation_eval_transforms(resize_size=image_size)
    train_dataset, val_dataset, test_dataset = make_datasets(train_dataset_str=train_dataset_str, val_dataset_str=val_dataset_str,
                                                            test_dataset_str=test_dataset_str, train_transform=train_image_transform,
                                                            eval_transform=eval_image_transform, train_target_transform=train_target_transform,
                                                            eval_target_transform=eval_target_transform)
    if shots != None:
        logger.info(f"Running dataset in {shots}-shot setting")
        train_dataset = FewShotDatasetWrapper(train_dataset, shots=shots)

    patch_size = model.patch_size
    batch_size = train_dataset.__len__() if batch_size > train_dataset.__len__() else batch_size
    embed_dim = model.embed_dim
    is_3d = test_dataset.is_3d()
    collate_fn = None if not is_3d else collate_fn_3d
    num_of_classes = test_dataset.get_num_classes()
    decoders, optim_param_groups = setup_decoders(
        embed_dim,
        learning_rates,
        num_of_classes,
        decoder_type,
        is_3d,
        image_size=image_size,
        patch_size=patch_size
    )

    if epoch_length == None:
        epoch_length = math.ceil(len(train_dataset) / batch_size)
    eval_period_epochs_ = eval_period_epochs * epoch_length
    checkpoint_period = save_checkpoint_frequency * epoch_length

    # Define feature model
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    n_last_blocks = 5 if decoder_type == "unet" else 1 
    feature_model = DINOV2Encoder(model, autocast_ctx=autocast_ctx, n_last_blocks=n_last_blocks, is_3d=is_3d)

    # Define checkpoint, optimizer, and scheduler
    optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
    if val_epochs is not None:
        max_iter = epoch_length * val_epochs
    else:
        max_iter = epoch_length * epochs 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    checkpointer = Checkpointer(decoders, output_dir, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(segmentor_fpath or "", resume=resume).get("iteration", 0) + 1
    if loss_function == "combined":
        loss_function = DiceCELoss(softmax=True, to_onehot_y=True)
        logging.info("Using combined dice and crossentropy loss")
    else:
        loss_function = DiceLoss(softmax=True, to_onehot_y=True)
        logging.info("Using dice loss")

    # Make dataloaders.
    sampler_type = SamplerType.INFINITE
    train_data_loader, val_data_loader, test_data_loader = make_data_loaders(train_dataset=train_dataset, test_dataset=test_dataset,
                                                                            val_dataset=val_dataset, sampler_type=sampler_type, seed=seed,
                                                                            start_iter=start_iter, batch_size=batch_size, num_workers=num_workers,
                                                                            collate_fn=collate_fn)

    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")
    val_results_dict, feature_model, decoders, iteration = eval_decoders(
        feature_model=feature_model,
        decoders=decoders,
        train_data_loader=train_data_loader,
        val_data_loader=test_data_loader if val_data_loader == None else val_data_loader,
        metrics_file_path=metrics_file_path,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        max_iter=max_iter,
        checkpoint_period=checkpoint_period,
        running_checkpoint_period=checkpoint_period//2,
        eval_period=eval_period_epochs_,
        metric_type=val_metric_type,
        num_of_classes=num_of_classes,
        resume=resume,
        segmentor_fpath=segmentor_fpath,
        is_3d=is_3d,
        loss_function=loss_function
    )

    if val_dataset != None: # retrain model with validation set.

        start_iter = 1

        if shots == None: # If few-shot is enabled, keep training set. 
            val_dataset = make_dataset(
                dataset_str=val_dataset_str,
                transform=train_image_transform,
                target_transform=train_target_transform
            )
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
            logger.info("Retraining model with combined dataset from train and validation")

        epoch_length = math.ceil(len(train_dataset) / batch_size)
        eval_period_epochs_ = eval_period_epochs * epoch_length
        checkpoint_period = save_checkpoint_frequency * epoch_length

        train_data_loader = make_data_loader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            seed=seed,
            sampler_type=sampler_type,
            sampler_advance=start_iter-1,
            drop_last=False,
            persistent_workers=False,
        )
        logger.info("Using the most optimal hp")
        hyperparameters = extract_hyperparameters_from_model(val_results_dict["best_segmentor"]["name"])
        learning_rate = hyperparameters["lr"]
      
        decoders, optim_param_groups = setup_decoders(
            embed_dim,
            learning_rate,
            num_of_classes,
            decoder_type,
            is_3d=is_3d,
            image_size=image_size,
            patch_size=patch_size
        )

        output_dir += os.sep + 'optimal'
        os.makedirs(output_dir, exist_ok=True)

        optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
        max_iter = epochs * epoch_length
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
        checkpointer = Checkpointer(decoders, output_dir, optimizer=optimizer, scheduler=scheduler)

        val_results_dict, feature_model, decoders, iteration = eval_decoders(
            feature_model=feature_model,
            decoders=decoders,
            train_data_loader=train_data_loader,
            val_data_loader=test_data_loader,
            metrics_file_path=metrics_file_path,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            max_iter=max_iter,
            checkpoint_period=checkpoint_period,
            running_checkpoint_period=checkpoint_period//2,
            eval_period=eval_period_epochs_,
            metric_type=val_metric_type,
            num_of_classes=num_of_classes,
            resume=resume,
            segmentor_fpath=segmentor_fpath,
            is_3d=is_3d,
            loss_function=loss_function
        )

    results_dict = {}
    results_dict["best_segmentor"] = val_results_dict["best_segmentor"]
    logger.info("Test Results Dict " + str(results_dict))

    test_dataset_name = test_dataset_str.split(":")[0]
    if test_dataset_name == "BTCV":
        save_test_results(feature_model=feature_model, 
                          decoder=decoders.module.decoders_dict[results_dict["best_segmentor"]],
                          dataset=test_dataset,
                          output_dir=output_dir)

    return results_dict

def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    run = partial(run_eval_segmentation,
        model=model,
        decoder_type=args.decoder_type,
        train_dataset_str=args.train_dataset_str,
        test_dataset_str=args.test_dataset_str,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_epochs=args.val_epochs,
        epoch_length=args.epoch_length,
        num_workers=args.num_workers,
        save_checkpoint_frequency=args.save_checkpoint_frequency,
        eval_period_epochs=args.eval_period_epochs,
        learning_rates=args.learning_rates,
        autocast_dtype=autocast_dtype,
        val_dataset_str=args.val_dataset_str,
        resume=not args.no_resume,
        segmentor_fpath=args.segmentor_fpath,
        val_metric_type=args.val_metric_type,
        image_size=args.image_size,
        loss_function=args.loss_function,
        backbone=args.backbone
    )
    if args.shots != None:
        for shot in args.shots:
            fs_output_dir = args.output_dir + os.sep + f"shots-{shot}"
            os.makedirs(fs_output_dir, exist_ok=True)
            run(shots=shot, output_dir=fs_output_dir)
    else:
        run(shots=args.shots, output_dir=args.output_dir,)
    return 0

# output_dir=args.output_dir,

if __name__ == "__main__":
    description = "Segmentation evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))