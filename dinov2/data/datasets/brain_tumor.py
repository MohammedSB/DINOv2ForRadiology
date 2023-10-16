import csv
from enum import Enum
import logging
import os
import shutil
import math
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Union
from torchvision.datasets import VisionDataset
from .medical_dataset import MedicalVisionDataset
from sklearn import preprocessing

import torch
import skimage
import pandas as pd
import numpy as np
import nibabel as nib
import h5py

class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 1344,
            _Split.VAL: 718,
            _Split.TEST: 1002,
        }
        return split_lengths[self]

class BrainTumor(MedicalVisionDataset):
    Split = _Split

    def __init__(
        self,
        *,
        split: "BrainTumor.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(split, root, transforms, transform, target_transform)

        self.class_names = np.array(["meningioma", "glioma", "pituitary"])

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def is_3d(self) -> bool:
        return False
    
    def is_multilabel(self) -> bool:
        return False

    def get_image_data(self, index: int) -> np.ndarray:
        image_path = self._split_dir + os.sep + self.images[index]
        file = h5py.File(image_path,'r')
        image = file.get('cjdata/image')
        image = np.stack((image,)*3, axis=0)
        
        # pre-preprocess
        max_value = np.percentile(image, 95)
        min_value = np.percentile(image, 5)
        image = np.where(image <= max_value, image, max_value)
        image = np.where(image <= min_value, 0., image)

        image = torch.tensor(image).float()

        return image
    
    def get_target(self, index: int) -> Tuple[np.ndarray, torch.Tensor, None]:
        label_path = self._split_dir + os.sep + self.images[index]
        file = h5py.File(label_path,'r')
        target = file.get('cjdata/label')
        target = torch.tensor(target).squeeze().type(torch.LongTensor) - 1
        return target
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):

        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transform is not None:
            image = self.transform(image)
        return image, target

def make_splits(data_dir = "/mnt/z/data/BrainTumor"):
    ids_to_files = OrderedDict()
    for i in range(1, 3065):
        file_name = f"{i}.mat"
        f = h5py.File(f'{data_dir}/{file_name}','r')
        data = f.get('cjdata/PID')
        data = np.array(data).flatten()
        data = ''.join(chr(value) for value in data)
        if data in ids_to_files.keys():
            ids_to_files[data].append(file_name)
        else:
            ids_to_files[data] = [file_name]

    n_patients = len(ids_to_files)
    test_list = np.arange(0, n_patients, n_patients/70).round().astype("int")
    val_list = np.arange(0, n_patients-70, (n_patients-70)/50).round().astype("int")

    os.makedirs(data_dir + "/test", exist_ok=True)
    os.makedirs(data_dir + "/val", exist_ok=True)
    os.makedirs(data_dir + "/train", exist_ok=True)

    to_path = data_dir + "/test"
    keys_to_remove = []
    for index in test_list:
        key_at_index = list(ids_to_files.keys())[index]
        for file in ids_to_files[key_at_index]:
            shutil.move(data_dir+f"/{file}", to_path)
        keys_to_remove.append(key_at_index)
    for key in keys_to_remove: ids_to_files.pop(key)

    to_path = data_dir + "/val"
    keys_to_remove = []
    for index in val_list:
        key_at_index = list(ids_to_files.keys())[index]
        for file in ids_to_files[key_at_index]:
            shutil.move(data_dir+f"/{file}", to_path)
        keys_to_remove.append(key_at_index)
    for key in keys_to_remove: ids_to_files.pop(key)

    to_path = data_dir + "/train"
    for index in ids_to_files:
        for file in ids_to_files[index]:
            shutil.move(data_dir+f"/{file}", to_path)