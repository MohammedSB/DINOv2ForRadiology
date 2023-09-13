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

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform
import dinov2.distributed as distributed
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import (ModelWithIntermediateLayers, evaluate, apply_method_to_nested_values,
                                make_datasets, make_data_loaders, extract_hyperparameters_from_model,
                                is_zero_matrix, collate_fn_3d)
from dinov2.eval.classification.utils import classifier_forward_pass
from dinov2.logging import MetricLogger


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
        help="Test dataset",
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
        help="Number of epochs between two evaluations.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--n-last-blocks",
        nargs="+",
        type=int
    )
    parser.add_argument(
        "--avgpools",
        nargs="+",
        type=bool
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
    parser.set_defaults(
        train_dataset_str="NIHChestXray:split=TRAIN",
        val_dataset_str=None,
        test_dataset_str="NIHChestXray:split=TEST",
        epochs=10,
        val_epochs=None,
        batch_size=128,
        num_workers=8,
        epoch_length=None,
        save_checkpoint_frequency=5,
        eval_period_epochs=5,
        learning_rates=[1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        n_last_blocks=[1,4],
        avgpools=[True, False],
        val_metric_type=MetricType.MULTILABEL_AUROC,
        classifier_fpath=None,
    )
    return parser


def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = torch.stack( # If 3D, take average of all slices.
            [create_linear_input(o, self.use_n_blocks, self.use_avgpool) for o in x_tokens_list]
            ).mean(dim=0)
        return self.linear(output)


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))

    def forward(self, samples, targets):
        preds = torch.sigmoid(self.linear_classifier(samples))
        return {
            "preds": preds[:, self.class_mapping] if self.class_mapping is not None else preds,
            "target": targets,
        }


def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size * distributed.get_global_size()) / 256.0


def setup_linear_classifiers(sample_output, n_last_blocks_list, learning_rates, avgpools=[True, False], num_classes=14):
    """
    Sets up the multiple linear classifiers with different hyperparameters to test out the most optimal one 
    """
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in avgpools:
            for _lr in learning_rates:
                # lr = scale_lr(_lr, batch_size)
                lr = _lr
                out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )
                linear_classifier = linear_classifier.cuda()
                linear_classifiers_dict[
                    f"linear:blocks={n}:avgpool={avgpool}:lr={lr:.10f}".replace(".", "_")
                ] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    if distributed.is_enabled():
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

    return linear_classifiers, optim_param_groups


@torch.no_grad()
def evaluate_linear_classifiers(
    feature_model,
    linear_classifiers,
    data_loader,
    metric_type,
    metrics_file_path,
    training_num_classes,
    iteration,
    prefixstring="",
    best_classifier_on_val=None,
):
    logger.info("running validation !")

    num_classes = training_num_classes
    labels = list(data_loader.dataset.class_names)
    metric = build_metric(metric_type, num_classes=num_classes, labels=labels)
    postprocessors = {k: LinearPostprocessor(v, None) for k, v in linear_classifiers.classifiers_dict.items()}
    metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}

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
    best_classifier = ""
    eval_metric = str(list(metric)[0])

    for i, (classifier_string, metric) in enumerate(results_dict_temp.items()):
        logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric}")
        if (
            best_classifier_on_val is None and metric[eval_metric].item() > max_score
        ) or classifier_string == best_classifier_on_val:
            max_score = metric[eval_metric].item()
            best_classifier = classifier_string

    results_dict["best_classifier"] = {"name": best_classifier, "results": apply_method_to_nested_values(
                                                                            results_dict_temp[best_classifier],
                                                                            method_name="item",
                                                                            nested_types=(dict))}

    logger.info(f"best classifier: {results_dict['best_classifier']}") 

    if distributed.is_main_process():
        with open(metrics_file_path, "a") as f:
            f.write(f"iter: {iteration}\n")
            for k, v in results_dict.items():
                f.write(json.dumps({k: v}) + "\n")
            f.write("\n")

    return results_dict


def eval_linear(
    *,
    feature_model,
    linear_classifiers,
    train_data_loader,
    val_data_loader,
    metrics_file_path,
    optimizer,
    scheduler,
    output_dir,
    max_iter,
    checkpoint_period,  # In number of epochs, creates a new file every period
    running_checkpoint_period,  # Period to update main checkpoint file
    eval_period,
    metric_type,
    training_num_classes,
    resume=True,
    classifier_fpath=None,
    is_multilabel=True,
    is_3d = False,
):
    checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", 0) + 1

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
        labels = labels.cuda(non_blocking=True)

        # forward pass
        outputs = classifier_forward_pass(feature_model, linear_classifiers, data=data, is_3d=is_3d)
        
        # calculate loss
        if is_multilabel:  
            losses = {}
            batch_size = labels.shape[0]
            for k, v in outputs.items():
                per_class_loss = torch.tensor([0.0], device=torch.cuda.current_device())
                for batch_index in range(batch_size): # Loop through each batch
                    batch_predictions = v[batch_index]
                    batch_labels = labels[batch_index]
                    for index, class_ in enumerate(batch_predictions): # Loop through each class prediciton
                        per_class_loss += nn.BCEWithLogitsLoss()(class_.float(), batch_labels[index].float())
                    losses[f"loss_{k}"] = per_class_loss / len(batch_labels) # Take average of all binary classification losses        
        else:
            loss_fn = nn.BCEWithLogitsLoss() if training_num_classes == 1 else nn.CrossEntropyLoss()
            losses = {f"loss_{k}": loss_fn(v, labels.float()) for k, v in outputs.items()}        

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
            _ = evaluate_linear_classifiers(
                feature_model=feature_model,
                linear_classifiers=remove_ddp_wrapper(linear_classifiers),
                data_loader=val_data_loader,
                metrics_file_path=metrics_file_path,
                prefixstring=f"ITER: {iteration}",
                metric_type=metric_type,
                training_num_classes=training_num_classes,
                iteration=iteration,
            )
            torch.cuda.synchronize()

        iteration = iteration + 1

    val_results_dict = evaluate_linear_classifiers(
        feature_model=feature_model,
        linear_classifiers=remove_ddp_wrapper(linear_classifiers),
        data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        metric_type=metric_type,
        training_num_classes=training_num_classes,
        iteration=iteration,
    )
    return val_results_dict, feature_model, linear_classifiers, iteration


def make_eval_data_loader(test_dataset_str, batch_size, num_workers, metric_type):
    test_dataset = make_dataset(
        dataset_str=test_dataset_str,
        transform=make_classification_eval_transform(),
    )
    test_data_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=None,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
        collate_fn=None,
    )
    return test_data_loader

def run_eval_linear(
    model,
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
    n_last_blocks_list,
    avgpools,
    autocast_dtype,
    val_dataset_str=None,
    resume=True,
    classifier_fpath=None,
    val_metric_type=MetricType.MULTILABEL_AUROC,
):
    seed = 0
    torch.manual_seed(seed)

    if test_dataset_str == None:
        raise ValueError("Test dataset cannot be None")
    
    n_last_blocks = max(n_last_blocks_list)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)

    train_transform = make_classification_train_transform()
    eval_transform = make_classification_eval_transform()
    train_dataset, val_dataset, test_dataset = make_datasets(train_dataset_str=train_dataset_str, val_dataset_str=val_dataset_str,
                                                        test_dataset_str=test_dataset_str, train_transform=train_transform,
                                                        eval_transform=eval_transform)
    training_num_classes = test_dataset.get_num_classes()
    training_num_classes = 1 if training_num_classes == 2 else training_num_classes
    is_multilabel = test_dataset.is_multilabel()
    is_3d = test_dataset.is_3d()
    collate_fn = None if not is_3d else collate_fn_3d

    sample_input = train_dataset[0][0][0] if is_3d else train_dataset[0][0] 
    sample_output = feature_model(sample_input.unsqueeze(0).cuda())


    if epoch_length == None:
        epoch_length = math.ceil(train_dataset.get_length() / batch_size)
    eval_period_epochs_ = eval_period_epochs * epoch_length
    checkpoint_period = save_checkpoint_frequency * epoch_length
    
    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output=sample_output,
        n_last_blocks_list=n_last_blocks_list,
        learning_rates=learning_rates,
        avgpools=avgpools,
        num_classes=training_num_classes,
    )

    optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
    if val_epochs is not None:
        max_iter = epoch_length * val_epochs
    else:
        max_iter = epoch_length * epochs 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", 0) + 1

    sampler_type = SamplerType.INFINITE
    train_data_loader, val_data_loader, test_data_loader = make_data_loaders(train_dataset=train_dataset, test_dataset=test_dataset,
                                                                        val_dataset=val_dataset, sampler_type=sampler_type, seed=seed,
                                                                        start_iter=start_iter, batch_size=batch_size, num_workers=num_workers,
                                                                        collate_fn=collate_fn)

    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")
    val_results_dict, feature_model, linear_classifiers, iteration = eval_linear(
        feature_model=feature_model,
        linear_classifiers=linear_classifiers,
        train_data_loader=train_data_loader,
        val_data_loader=test_data_loader if val_data_loader is None else val_data_loader,
        metrics_file_path=metrics_file_path,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        max_iter=max_iter,
        checkpoint_period=checkpoint_period,
        running_checkpoint_period=epoch_length,
        eval_period=eval_period_epochs_,
        metric_type=val_metric_type,
        training_num_classes=training_num_classes,
        resume=resume,
        classifier_fpath=classifier_fpath,
        is_multilabel=is_multilabel,
        is_3d=is_3d
    )

    if val_dataset_str != None: # retrain model with validation set.

        start_iter = 1

        val_dataset = make_dataset(
            dataset_str=val_dataset_str,
            transform=train_transform,
        )
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

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
        logger.info("Retraining model with combined dataset from train and validation, using the most optimal hp.")
        hyperparameters = extract_hyperparameters_from_model(val_results_dict["best_classifier"]["name"])
        learning_rate, avgpool, block = hyperparameters["lr"], hyperparameters["avgpool"], hyperparameters["blocks"]
      
        linear_classifiers, optim_param_groups = setup_linear_classifiers(
            sample_output=sample_output,
            n_last_blocks_list=block,
            learning_rates=learning_rate,
            avgpools=avgpool,
            num_classes=training_num_classes,
        )

        output_dir += os.sep + 'optimal'
        os.makedirs(output_dir, exist_ok=True)

        optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
        max_iter = epochs * epoch_length
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
        checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)

        val_results_dict, feature_model, linear_classifiers, iteration = eval_linear(
            feature_model=feature_model,
            linear_classifiers=linear_classifiers,
            train_data_loader=train_data_loader,
            val_data_loader=test_data_loader,
            metrics_file_path=metrics_file_path,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            max_iter=max_iter,
            checkpoint_period=checkpoint_period,
            running_checkpoint_period=epoch_length,
            eval_period=eval_period_epochs_,
            metric_type=val_metric_type,
            training_num_classes=training_num_classes,
            resume=resume,
            classifier_fpath=classifier_fpath,
            is_multilabel=is_multilabel,
            is_3d=is_3d
        )

    results_dict = {}
    results_dict["best_classifier"] = val_results_dict["best_classifier"]
    logger.info("Test Results Dict " + str(results_dict))

    return results_dict


def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    run_eval_linear(
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        test_dataset_str=args.test_dataset_str,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_epochs=args.val_epochs,
        epoch_length=args.epoch_length,
        num_workers=args.num_workers,
        save_checkpoint_frequency=args.save_checkpoint_frequency,
        eval_period_epochs=args.eval_period_epochs,
        learning_rates=args.learning_rates,
        n_last_blocks_list=args.n_last_blocks,
        avgpools=args.avgpools,
        autocast_dtype=autocast_dtype,
        resume=not args.no_resume,
        classifier_fpath=args.classifier_fpath,
        val_metric_type=args.val_metric_type,
    )
    return 0


if __name__ == "__main__":
    description = "DINOv2 linear evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))