import csv
import logging
import os
import shutil
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import skimage
import torch
from sklearn import preprocessing
from torchvision.datasets import VisionDataset
from .medical_dataset import MedicalVisionDataset

logger = logging.getLogger("dinov2")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 191_027,
            _Split.VAL: 202,
            _Split.TEST: 518,
        }
        return split_lengths[self]

class CheXpert(MedicalVisionDataset):
    Split = _Split

    def __init__(
        self,
        *,
        split: "CheXpert.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        root = root + os.sep
        super().__init__(split, root, transforms, transform, target_transform)
        
        # Set the labels dataframe
        self.root = root + os.sep
        self.labels = pd.read_csv(self.root + self._split.value + ".csv")
        self._clean_labels()

    def _check_size(self):
        t = pd.read_csv(self.root + self._split.value + ".csv")
        t= t[~t['Path'].str.contains('lateral')].reset_index(drop=True)
        num_of_images = len(t)
        logger.info(f"{self._split.length - num_of_images} scans are missing from {self._split.value.upper()} set")

    def _clean_labels(self):

        self.labels = self.labels[~self.labels['Path'].str.contains('lateral')].reset_index(drop=True)
        self.labels.fillna(0, inplace=True)
        self.labels = self.labels[["Path", "Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]]
        self.labels["Uncertain"] = 0

        # Loop through the rows of the table
        for i, row in self.labels.iterrows():
            # Check if any of the diseases have value -1
            if any(row[["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]] == -1):
                # Set uncertain to 1
                self.labels.loc[i, "Uncertain"] = 1
        self.labels.replace(-1, 0, inplace=True)

        classes = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion", "Uncertain"]
        self.targets = self.labels[classes].to_numpy()
        self.class_names = classes

    @property
    def split(self) -> "CheXpert.Split":
        return self._split
    
    def get_length(self) -> int:
        return self.__len__()
    
    def get_num_classes(self) -> int:
        return len(self.class_names)

    def is_3d(self) -> bool:
        return False
    
    def is_multilabel(self) -> bool:
        return True

    def get_image_data(self, index: int) :
        data_point = self.labels.iloc[index] 
        rel = f"{os.sep}".join(data_point["Path"].split(f"{os.sep}")[1:]) if self._split == _Split.TEST else \
                f"{os.sep}".join(data_point["Path"].split(f"{os.sep}")[2:])
        image_path = self._split_dir + os.sep + rel
        
        # Read as gray because some of the images have extra layers in the 3rd dimension
        image = skimage.io.imread(image_path).astype(np.float16)
        image = np.stack((image,)*3, axis=0)
        image = torch.from_numpy(image)

        return image

    def get_target(self, index: int):
        return self.targets[index].astype(np.int64)

    def get_targets(self) -> np.ndarray:
        return self.targets
    
    def __getitem__(self, index):
        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.labels)
    
def make_val_set(data_dir="/mnt/d/data/tmp/CheXpert-v1.0/"):
    df = pd.read_csv(data_dir + "train.csv")
    df = df[~df['Path'].str.contains('lateral')].reset_index(drop=True)
    # df = df[df["AP/PA"] == "AP"].reset_index(drop=True)
    df.fillna(0, inplace=True)
    df = df[["Path", "Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]]

    df["Uncertain"] = 0

    # Loop through the rows of the table
    for i, row in df.iterrows():
        # Check if any of the diseases have value -1
        if any(row[["Cardiomegaly", "Eadema", "Consolidation", "Atelectasis", "Pleural Effusion"]] == -1):
            # Set uncertain to 1
            df.loc[i, "Uncertain"] = 1

    df.replace(-1, 0, inplace=True)

    length = len(df)
    val_indices = np.arange(0, length, length/10_000).round().astype("int")
    df_val = df.iloc[val_indices]
    df_train = df.drop(val_indices)

    df_train.to_csv(data_dir + "train.csv", index=False)
    df_val.to_csv(data_dir + "val.csv", index=False)