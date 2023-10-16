# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
import logging
from typing import Any, Dict, Optional
from collections import OrderedDict

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import (MultilabelAUROC, MultilabelF1Score, MultilabelAccuracy, MulticlassF1Score, 
                                        MulticlassAccuracy, MulticlassAUROC, Accuracy, BinaryF1Score, BinaryAUROC,
                                        JaccardIndex, MulticlassJaccardIndex, Dice, BinaryAUROC)
from torchmetrics.utilities.data import dim_zero_cat, select_topk

logger = logging.getLogger("dinov2")

class MetricType(Enum):
    MEAN_ACCURACY = "mean_accuracy"
    MEAN_PER_CLASS_ACCURACY = "mean_per_class_accuracy"
    MULTILABEL_ACCURACY = "multilabel_accuracy"
    MULTILABEL_AUROC = "multilabel_auc"
    MULTICLASS_AUROC = "multiclass_auc"
    BINARY_AUROC = "binary_auc"
    PER_CLASS_ACCURACY = "per_class_accuracy"
    IMAGENET_REAL_ACCURACY = "imagenet_real_accuracy"

    SEGMENTATION_METRICS = "segmentation_metrics"

    @property
    def accuracy_averaging(self):
        return getattr(MetricAveraging, self.name, None)

    def __str__(self):
        return self.value


class MetricAveraging(Enum):
    MEAN_ACCURACY = "micro"
    MEAN_PER_CLASS_ACCURACY = "macro"
    MULTILABEL_ACCURACY = "macro"
    MULTILABEL_AUROC = "macro"
    MULTICLASS_AUROC = "macro"
    BINARY_AUROC = "macro"
    MULTCLASS_JACCARD = "macro"
    PER_CLASS_ACCURACY = "none"

    SEGMENTATION_METRICS = "macro"

    def __str__(self):
        return self.value


def build_metric(metric_type: MetricType, *, num_classes: int, labels = None, ks: Optional[tuple] = None):
    if metric_type == MetricType.MULTILABEL_ACCURACY:
        return build_multilabel_accuracy_metric(
            average_type=metric_type.accuracy_averaging,
            num_labels=num_classes,
            labels=labels
        )
    elif metric_type == MetricType.MULTILABEL_AUROC:
        return build_multilabel_auroc_metric(
            average_type=metric_type.accuracy_averaging,
            num_labels=num_classes,
            labels=labels
        )
    elif metric_type == MetricType.MULTICLASS_AUROC:
        return build_multiclass_auroc_metric(
            average_type=metric_type.accuracy_averaging,
            num_classes=num_classes,
            labels=labels
        )
    elif metric_type == MetricType.SEGMENTATION_METRICS:
        return build_segmentation_metrics(
            average_type=metric_type.accuracy_averaging,
            num_labels=num_classes,
            labels=labels
        )
    elif metric_type == MetricType.BINARY_AUROC:
        return build_binary_auroc_metric()

    if metric_type.accuracy_averaging is not None:
        return build_topk_accuracy_metric(
            average_type=metric_type.accuracy_averaging,
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )
    elif metric_type == MetricType.IMAGENET_REAL_ACCURACY:
        return build_topk_imagenet_real_accuracy_metric(
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )

    raise ValueError(f"Unknown metric type {metric_type}")


def build_segmentation_metrics(average_type: MetricAveraging, num_labels: int=2, labels=None):
    metrics: Dict[str, Metric] = {
        "jaccard": MulticlassJaccardIndex(num_classes=num_labels, average=average_type.value, ignore_index=0),
        "dice": Dice(num_classes=num_labels, average=average_type.value, ignore_index=0),
    }
    return MetricCollection(metrics)

def build_multilabel_accuracy_metric(average_type: MetricAveraging, num_labels: int):
    metrics: Dict[str, Metric] = {
         f"accuracy": MultilabelAccuracy(num_labels=num_labels, average=average_type.value)
    }
    return MetricCollection(metrics)

def build_multilabel_auroc_metric(average_type: MetricAveraging, num_labels: int, labels=None):
    metrics: Dict[str, Metric] = {
        "auroc": MultilabelAUROC(num_labels=num_labels, average=average_type.value),
        "class-specific": MetricCollection({
            "auroc": ClasswiseWrapper(MultilabelAUROC(num_labels=num_labels, average=None), labels=labels, prefix="_"),
        })
    }
    return MetricCollection(metrics)

def build_multiclass_auroc_metric(average_type: MetricAveraging, num_classes: int, labels=None):
    metrics: Dict[str, Metric] = {
        "auroc": MulticlassAUROC(num_classes=num_classes, average=average_type.value),
        "class-specific": MetricCollection({
            "auroc": ClasswiseWrapper(MulticlassAUROC(num_classes=num_classes, average=None), labels=labels, prefix="_"),
        })
    }
    return MetricCollection(metrics)

def build_binary_auroc_metric():
    metrics: Dict[str, Metric] = {
        "auroc": BinaryAUROC()
        }
    return MetricCollection(metrics)

def build_multilabel_metrics(average_type: MetricAveraging, num_labels: int, labels=None):
    metrics: Dict[str, Metric] = {
        "auroc": MultilabelAUROC(num_labels=num_labels, average=average_type.value),
        "accuracy": MultilabelAccuracy(num_labels=num_labels, average=average_type.value),
        "f1": MultilabelF1Score(num_labels=num_labels, average=average_type.value),
        "class-specific": MetricCollection({
            "auroc": ClasswiseWrapper(MultilabelAUROC(num_labels=num_labels, average=None), labels=labels, prefix="_"),
            "accuracy":ClasswiseWrapper(MultilabelAccuracy(num_labels=num_labels, average=None), labels=labels, prefix="_"),
            "f1": ClasswiseWrapper(MultilabelF1Score(num_labels=num_labels, average=None), labels=labels, prefix="_")
        })
    }
    return MetricCollection(metrics)
    

def build_topk_accuracy_metric(average_type: MetricAveraging, num_classes: int, ks: tuple = (1, 5)):
    metrics: Dict[str, Metric] = {
        f"top-{k}": MulticlassAccuracy(top_k=k, num_classes=int(num_classes), average=average_type.value) for k in ks
    }
    return MetricCollection(metrics)


def build_topk_imagenet_real_accuracy_metric(num_classes: int, ks: tuple = (1, 5)):
    metrics: Dict[str, Metric] = {f"top-{k}": ImageNetReaLAccuracy(top_k=k, num_classes=int(num_classes)) for k in ks}
    return MetricCollection(metrics)
    

class ImageNetReaLAccuracy(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.top_k = top_k
        self.add_state("tp", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        # preds [B, D]
        # target [B, A]
        # preds_oh [B, D] with 0 and 1
        # select top K highest probabilities, use one hot representation
        preds_oh = select_topk(preds, self.top_k)
        # target_oh [B, D + 1] with 0 and 1
        target_oh = torch.zeros((preds_oh.shape[0], preds_oh.shape[1] + 1), device=target.device, dtype=torch.int32)
        target = target.long()
        # for undefined targets (-1) use a fake value `num_classes`
        target[target == -1] = self.num_classes
        # fill targets, use one hot representation
        target_oh.scatter_(1, target, 1)
        # target_oh [B, D] (remove the fake target at index `num_classes`)
        target_oh = target_oh[:, :-1]
        # tp [B] with 0 and 1
        tp = (preds_oh * target_oh == 1).sum(dim=1)
        # at least one match between prediction and target
        tp.clip_(max=1)
        # ignore instances where no targets are defined
        mask = target_oh.sum(dim=1) > 0
        tp = tp[mask]
        self.tp.append(tp)  # type: ignore

    def compute(self) -> Tensor:
        tp = dim_zero_cat(self.tp)  # type: ignore
        return tp.float().mean()
