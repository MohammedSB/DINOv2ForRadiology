# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union
from torchvision.datasets import VisionDataset
from sklearn import preprocessing

import torch
import skimage
import pandas as pd
import numpy as np


logger = logging.getLogger("dinov2")
_Target = int

class NIHChestXray(VisionDataset):
    def __init__(
        self,
        *,
        split: str,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        
        # Define paths for the data
        self._root = root
        self.labels_path = self._root + "labels"
        self.images_path = self._root + split

        self.split = split
        self.labels_df = pd.read_csv(self.labels_path + ".csv")

        self._extract_subset(split)

    def _clean_labels(self):
        # Define inner split string function
        def spilt_string(string):
            splitted = string.split("|")
            return splitted

        # Turn all labels into list
        self.labels_df["Finding Labels"] = self.labels_df["Finding Labels"].apply(spilt_string)

        # Encoding of multilabeled targets
        mlb = preprocessing.MultiLabelBinarizer()
        targets = mlb.fit_transform(self.labels_df["Finding Labels"])
        self.class_names = mlb.classes_
        self.targets = pd.DataFrame(targets, columns=mlb.classes_)

    def _extract_subset(self, split):
        # Define either train or testset
        if split == "train" or split == "val":
            subset = pd.read_csv(self.data_location + "train_val_list.txt", names=["Image Index"])
        elif split == "test":
            subset = pd.read_csv(self.data_location + "test_list.txt", names=["Image Index"])
        else:
            raise ValueError(f'Unsupported split "{split}"')

        self.labels_df = pd.merge(self.labels_df, subset, how="inner", on=["Image Index"])
        self._clean_labels()

    @property
    def split(self) -> str:
        return self.split

    def find_class_id(self, class_index: int) -> str:
        class_ids = self._get_class_ids()
        return str(class_ids[class_index])

    def find_class_name(self, class_index: int) -> str:
        class_names = self._get_class_names()
        return str(class_names[class_index])

    def get_image_data(self, index: int) :
        data_point = self.labels_df.iloc[index]
        image_path = self.images_path + os.sep + data_point["Image Index"]

        # Read as gray because some of the images have extra layers in the 3rd dimension
        image = skimage.io.imread(image_path, as_gray=True).astype(np.float32)
        image = np.expand_dims(image, axis=0)
        image = self._transform(image)

        image = torch.from_numpy(image)
        return image

    def get_target(self, index: int):
        return None if self.split == "test" else self.targets.iloc[index]

    def get_targets(self) -> Optional[np.ndarray]:
        return None if self.split == "test" else self.targets

    def get_class_id(self, index: int) -> Optional[str]:
        class_id = self.targets[index]
        return None if self.split == "test" else str(class_id)

    def get_class_name(self, index: int) -> Optional[str]:
        class_name_index = self.targets[index]
        class_name = self.class_names[class_name_index]
        return None if self.split == "test" else str(class_name)
    
    def __getitem__(self, index):
        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.labels_df)