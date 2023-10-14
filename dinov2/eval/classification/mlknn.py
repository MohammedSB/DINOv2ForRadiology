# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional


import numpy as np
from scipy import sparse
from sklearn import multiclass
import sklearn.metrics 
from sklearn.datasets import make_multilabel_classification
from sklearn.neighbors import NearestNeighbors

from skmultilearn.utils import get_matrix_in_format

import torch
from torch.nn.functional import one_hot, softmax

import dinov2.distributed as distributed
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.eval.metrics import MetricCollection, MetricType, MetricAveraging, build_topk_accuracy_metric, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser, setup_and_build_model
from dinov2.eval.utils import ModelWithNormalize, MLkNN, evaluate, extract_features, apply_method_to_nested_values

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
        "--test-dataset",
        dest="test_dataset_str",
        type=str,
        help="Test dataset",
    )
    parser.add_argument(
        "--nb_knn",
        nargs="+",
        type=int,
        help="Number of NN to use. 20 is usually working the best.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature used in the voting coefficient",
    )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower"
        "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--n-per-class-list",
        nargs="+",
        type=int,
        help="Number to take per class",
    )
    parser.add_argument(
        "--n-tries",
        type=int,
        help="Number of tries",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        help="The name of the backbone model to use [dinov2, vit-large-imagenet21k]",
    )
    parser.set_defaults(
        train_dataset_str="NIHChestXray:split=TRAIN",
        test_dataset_str="NIHChestXray:split=TEST",
        nb_knn=[5],
        temperature=0.07,
        batch_size=16,
        n_per_class_list=[-1],
        n_tries=1,
        backbone="dinov2",
    )
    return parser


def eval_knn(
    model,
    train_dataset,
    test_dataset,
    nb_knn,
    batch_size,
    num_workers,
    gather_on_cpu,
    metric_type=MetricType.MULTILABEL_AUROC
):
    model = ModelWithNormalize(model)

    logger.info("Extracting features for train set...")
    train_features, train_labels = extract_features(
        model, train_dataset, batch_size, num_workers, gather_on_cpu=gather_on_cpu
    )
    logger.info(f"Train features created, shape {train_features.shape}.")

    model.eval()
    logger.info("Extracting features for evaluation set...")
    test_features, test_labels = extract_features(
        model, test_dataset, batch_size, num_workers, gather_on_cpu=gather_on_cpu
    )

    labels = list(test_dataset.class_names)
    num_classes = test_dataset.get_num_classes()

    train_features, train_labels = train_features.cpu().numpy(), train_labels.cpu().numpy()
    test_features, test_labels = test_features.cpu().numpy(), test_labels.cpu().numpy()

    results_dict = {}
    # ============ evaluation ... ============
    logger.info("Start the Multilabel k-NN classification.")
    for k in nb_knn:

        results_dict[f"{k}"] = {}

        classifier = MLkNN(k)
        classifier.fit(train_features, train_labels)
        results = torch.tensor(classifier.predict_proba(test_features).toarray()).cuda()
        
        metric = build_metric(metric_type, num_classes=num_classes, labels=labels)
        metric.update(**{"target": torch.tensor(test_labels).cuda(), "preds": results})
        metric.compute()

        results_dict[f"{k}"] = apply_method_to_nested_values(metric, "compute", nested_types=(MetricCollection, dict))

    results_dict = apply_method_to_nested_values(results_dict, "item")
    return results_dict


def eval_knn_with_model(
    model,
    output_dir,
    train_dataset_str="NIHChestXray:split=TRAIN",
    test_dataset_str="NIHChestXray:split=TEST",
    nb_knn=(5, 20, 50, 100, 200),
    autocast_dtype=torch.float,
    transform=None,
    gather_on_cpu=False,
    batch_size=256,
    num_workers=5,
):
    
    transform = transform or make_classification_eval_transform()

    train_dataset = make_dataset(
        dataset_str=train_dataset_str,
        transform=transform,
    )
    val_dataset = make_dataset(
        dataset_str=train_dataset_str.replace("TRAIN", "VAL"),
        transform=transform
    )
    train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

    test_dataset = make_dataset(
        dataset_str=test_dataset_str,
        transform=transform,
    )

    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        results_dict_knn = eval_knn(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            nb_knn=nb_knn,
            batch_size=batch_size,
            num_workers=num_workers,
            gather_on_cpu=gather_on_cpu,
        )

    metrics_file_path = os.path.join(output_dir, "results_eval_knn.json")
    with open(metrics_file_path, "a") as f:
        for k, v in results_dict_knn.items():
            f.write(json.dumps({k: v}) + "\n")

    if distributed.is_enabled():
        torch.distributed.barrier()
    return results_dict_knn


def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    eval_knn_with_model(
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        test_dataset_str=args.test_dataset_str,
        nb_knn=args.nb_knn,
        autocast_dtype=autocast_dtype,
        transform=None,
        gather_on_cpu=args.gather_on_cpu,
        batch_size=args.batch_size,
        num_workers=2,
    )
    return 0

if __name__ == "__main__":
    description = "DINOv2 Multilabel k-NN evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
