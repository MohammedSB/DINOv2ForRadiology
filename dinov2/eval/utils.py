# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional
from builtins import range

from sklearn.neighbors import NearestNeighbors
from skmultilearn.base import MLClassifierBase
from skmultilearn.utils import get_matrix_in_format

import numpy as np
import scipy.sparse as sparse

import torch
from torch import nn
from torchmetrics import MetricCollection

from dinov2.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader, make_dataset
import dinov2.distributed as distributed
from dinov2.logging import MetricLogger


logger = logging.getLogger("dinov2")


class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=1, p=2)


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx, is_3d=True, fine_tune=False):
        super().__init__()
        self.feature_model = feature_model
        self.fine_tune = fine_tune
        if self.fine_tune:
            self.feature_model.train()
        else:
            self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx
        self.is_3d = is_3d

    def forward_3d(self, images):
        batch_features = [] 
        for batch_scans in images: # calculate the features for every scan in all scans of the batch
            scans = []
            for scan in batch_scans:
                if not is_zero_matrix(scan): scans.append(self.forward_(scan.unsqueeze(0)))
            batch_features.append(scans)
        return batch_features

    def forward_(self, images):
        with self.autocast_ctx():
            if self.fine_tune:
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
            else:
                with torch.no_grad():
                    features = self.feature_model.get_intermediate_layers(
                        images, self.n_last_blocks, return_class_token=True
                    )
        return features
    
    def forward(self, images):
        if self.is_3d: return self.forward_3d(images)
        return [self.forward_(images)]


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
        
        outputs = model(samples.to(device))
        if isinstance(targets, torch.Tensor):
            targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = apply_method_to_nested_values(metrics, "compute", nested_types=(MetricCollection, dict))
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features(model, dataset, batch_size, num_workers, gather_on_cpu=False):
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets)
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=None,
        drop_last=False,
        shuffle=False,
    )
    return extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu)


@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(delimiter="  ")
    features, all_labels = None, None
    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        features_rank = model(samples).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    logger.info(f"Features shape: {tuple(features.shape)}")
    logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels


class MLkNN(MLClassifierBase):
    """kNN classification method adapted for multi-label classification
    References
    ----------
    If you use this classifier please cite the original paper introducing the method:
    .. code :: bibtex

        @article{zhang2007ml,
          title={ML-KNN: A lazy learning approach to multi-label learning},
          author={Zhang, Min-Ling and Zhou, Zhi-Hua},
          journal={Pattern recognition},
          volume={40},
          number={7},
          pages={2038--2048},
          year={2007},
          publisher={Elsevier}
        }

    """

    def __init__(self, k=10, s=1.0, ignore_first_neighbours=0, n_jobs=None, metric="cosine"):
        super(MLkNN, self).__init__()
        self.k = k  # Number of neighbours
        self.s = s  # Smooth parameter
        self.ignore_first_neighbours = ignore_first_neighbours
        self.n_jobs = n_jobs
        self.knn_ = NearestNeighbors(n_neighbors=self.k, metric=metric)
        self.copyable_attrs = ["k", "s", "ignore_first_neighbours", "n_jobs"]

    def _compute_prior(self, y):
        prior_prob_true = np.array(
            (self.s + y.sum(axis=0)) / (self.s * 2 + self._num_instances)
        )[0]
        prior_prob_false = 1 - prior_prob_true

        return (prior_prob_true, prior_prob_false)

    def _compute_cond(self, X, y):
        self.knn_.fit(X)
        c = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="i8")
        cn = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="i8")

        label_info = get_matrix_in_format(y, "dok")

        neighbors = [
            a[self.ignore_first_neighbours :]
            for a in self.knn_.kneighbors(
                X, self.k + self.ignore_first_neighbours, return_distance=False
            )
        ]

        for instance in range(self._num_instances):
            deltas = label_info[neighbors[instance], :].sum(axis=0)
            for label in range(self._num_labels):
                if label_info[instance, label] == 1:
                    c[label, deltas[0, label]] += 1
                else:
                    cn[label, deltas[0, label]] += 1

        c_sum = c.sum(axis=1)
        cn_sum = cn.sum(axis=1)

        cond_prob_true = sparse.lil_matrix(
            (self._num_labels, self.k + 1), dtype="float"
        )
        cond_prob_false = sparse.lil_matrix(
            (self._num_labels, self.k + 1), dtype="float"
        )
        for label in range(self._num_labels):
            for neighbor in range(self.k + 1):
                cond_prob_true[label, neighbor] = (self.s + c[label, neighbor]) / (
                    self.s * (self.k + 1) + c_sum[label, 0]
                )
                cond_prob_false[label, neighbor] = (self.s + cn[label, neighbor]) / (
                    self.s * (self.k + 1) + cn_sum[label, 0]
                )
        return cond_prob_true, cond_prob_false

    def fit(self, X, y):
        self._label_cache = get_matrix_in_format(y, "lil")
        self._num_instances = self._label_cache.shape[0]
        self._num_labels = self._label_cache.shape[1]
        # Computing the prior probabilities
        self._prior_prob_true, self._prior_prob_false = self._compute_prior(
            self._label_cache
        )
        # Computing the posterior probabilities
        self._cond_prob_true, self._cond_prob_false = self._compute_cond(
            X, self._label_cache
        )
        return self

    def predict(self, X):
        result = sparse.lil_matrix((X.shape[0], self._num_labels), dtype="i8")
        neighbors = [
            a[self.ignore_first_neighbours :]
            for a in self.knn_.kneighbors(
                X, self.k + self.ignore_first_neighbours, return_distance=False
            )
        ]
        for instance in range(X.shape[0]):
            deltas = self._label_cache[neighbors[instance],].sum(axis=0)

            for label in range(self._num_labels):
                p_true = (
                    self._prior_prob_true[label]
                    * self._cond_prob_true[label, deltas[0, label]]
                )
                p_false = (
                    self._prior_prob_false[label]
                    * self._cond_prob_false[label, deltas[0, label]]
                )
                result[instance, label] = int(p_true >= p_false)

        return result

    def predict_proba(self, X):
        result = sparse.lil_matrix((X.shape[0], self._num_labels), dtype="float")
        neighbors = [
            a[self.ignore_first_neighbours :]
            for a in self.knn_.kneighbors(
                X, self.k + self.ignore_first_neighbours, return_distance=False
            )
        ]
        for instance in range(X.shape[0]):
            deltas = self._label_cache[neighbors[instance],].sum(axis=0)

            for label in range(self._num_labels):
                p_true = (
                    self._prior_prob_true[label]
                    * self._cond_prob_true[label, deltas[0, label]]
                )
                p_false = (
                    self._prior_prob_false[label]
                    * self._cond_prob_false[label, deltas[0, label]]
                )
                result[instance, label] = p_true / (p_true + p_false)

        return result
    
def apply_method_to_nested_values(d, method_name, nested_types=(dict)):
    result = {}
    for key, value in d.items():
        if isinstance(value, nested_types):
            result[key] = apply_method_to_nested_values(value, method_name)
        else:
            method = getattr(value, method_name)
            result[key] = method()
    return result

def make_datasets(train_dataset_str, test_dataset_str, val_dataset_str=None,
                  train_transform=None, eval_transform=None, train_target_transform=None, eval_target_transform=None):
    train_dataset = make_dataset(
        dataset_str=train_dataset_str,
        transform=train_transform,
        target_transform=train_target_transform
    )
    if val_dataset_str == None:
        if train_dataset_str.replace("TRAIN", "VAL") != test_dataset_str:
            val_dataset_ = make_dataset(
                dataset_str=train_dataset_str.replace("TRAIN", "VAL"),
                transform=train_transform,
                target_transform=train_target_transform
            )
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset_])
        val_dataset = None
        logger.info("Train and val datasets have been combined.")
    else:
        val_dataset = make_dataset(
            dataset_str=val_dataset_str,
            transform=eval_transform,
            target_transform=eval_target_transform
        )
    test_dataset = make_dataset(
        dataset_str=test_dataset_str,
        transform=eval_transform,
        target_transform=eval_target_transform
    )
    return train_dataset, val_dataset, test_dataset

def make_data_loaders(train_dataset, test_dataset, val_dataset=None,
                    sampler_type=SamplerType.INFINITE, seed=0, start_iter=1,
                    batch_size=16, num_workers=0, collate_fn=None):
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
        collate_fn=collate_fn
    )
    val_data_loader = None
    if val_dataset != None:
        val_data_loader = make_data_loader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler_type=None, 
            drop_last=False,
            shuffle=False,
            persistent_workers=False,
            collate_fn=collate_fn,
        )
    test_data_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=None, 
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
        collate_fn=collate_fn,
    )

    return train_data_loader, val_data_loader, test_data_loader

def extract_hyperparameters_from_model(segmentor):
    hps = segmentor.split(":")[1:]
    hyperparameters = {}
    for hp in hps:
        key, value = hp.split("=")
        if key == "lr":
            value = float(value.replace("_", ".")) 
            hyperparameters[key] = [value]
        elif key == "avgpool":
            hyperparameters[key] = [bool(value)]
        elif key == "blocks":
            hyperparameters[key] = [int(value)]
        else:
            hyperparameters[key] = [value]
    return hyperparameters

def collate_fn_3d(batch):
    # batch is a list of tuples where each tuple is (video, label)
    videos, labels = zip(*batch)
    hw_size = videos[0].size()[-1]

    # Get the length of the longest video
    max_len = max(video.size(0) for video in videos)

    # Create a tensor to hold the padded videos
    padded_videos = torch.zeros(len(videos), max_len, 3, hw_size, hw_size)

    # Pad each video
    for i, video in enumerate(videos):
        padded_videos[i, :video.size(0)] = video

    return padded_videos, labels

def is_zero_matrix(matrix):
    return torch.allclose(matrix, torch.zeros_like(matrix))