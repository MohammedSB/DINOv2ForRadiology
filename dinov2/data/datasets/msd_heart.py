import csv
from enum import Enum
import logging
import os
import shutil
import math
from typing import Callable, List, Optional, Tuple, Union
from torchvision.datasets import VisionDataset
from .medical_dataset import MedicalVisionDataset
from sklearn import preprocessing

import torch
import skimage
import pandas as pd
import numpy as np
import nibabel as nib

class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 1339,
            _Split.VAL: 450,
            _Split.TEST: 482,
        }
        return split_lengths[self]

class MSDHeart(MedicalVisionDataset):
    Split = _Split

    def __init__(
        self,
        *,
        split: "MSDHeart.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(split, root, transforms, transform, target_transform)

        self.images = np.sort(np.array(os.listdir(self._split_dir)))

        self._labels_path = f"{os.sep}".join(self._split_dir.split(f"{os.sep}")[:-1]) + os.sep + "labels"
        self.labels = np.sort(np.array(os.listdir(self._labels_path)))
        
        labels_in = np.isin(self.labels, self.images)
        self.labels = self.labels[labels_in]
    
        self.class_id_mapping = pd.DataFrame([i for i in range(2)],
                                    index=["background", "left atrium"],
                                    columns=["class_id"])
        self.class_names = np.array(self.class_id_mapping.index)

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def is_3d(self) -> bool:
        return False

    def get_image_data(self, index: int) -> np.ndarray:
        image_path = self._split_dir + os.sep + self.images[index]

        image = np.load(image_path)
        image = np.stack((image,)*3, axis=0)
        image = torch.tensor(image).float()

        # pre-preprocess
        max_value = np.percentile(image, 95)
        min_value = np.percentile(image, 5)
        image = np.clip(image, min_value, max_value)

        return image
    
    def get_target(self, index: int) -> Tuple[np.ndarray, torch.Tensor, None]:        
        label_path = self._labels_path + os.sep + self.labels[index]
                
        label = np.load(label_path)
        label = torch.from_numpy(label).unsqueeze(0)

        return label
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):

        seed = np.random.randint(2147483647) # make a seed with numpy generator 

        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transform is not None:
            np.random.seed(seed), torch.manual_seed(seed) 
            image = self.transform(image)

        if self.target_transform is not None and target is not None:
            np.random.seed(seed), torch.manual_seed(seed) 
            target = self.target_transform(target)

        # Remove channel dim in target
        target = target.squeeze()

        return image, target

def slice_it(data_dir="/mnt/z/data/BTCVSlice"):
    train_data_path = data_dir + "train/"
    train_data = os.listdir(train_data_path)

    val_data_path = data_dir + "val/"
    val_data = os.listdir(val_data_path)

    test_data_path = data_dir + "test/"
    test_data = os.listdir(test_data_path)

    for scan in train_data:
        nifit_scan = nib.load(train_data_path + scan)
        array = nifit_scan.get_fdata()
        array = array.transpose(2, 0, 1)
        
        scan_name = scan.split(".")[0]
        for i, slice in enumerate(array):
            num = str(i).zfill(3)
            np.save(train_data_path + os.sep + scan_name  + f"_{num}.npy", slice)
        os.remove(train_data_path + scan)

    for scan in val_data:
        nifit_scan = nib.load(val_data_path + scan)
        array = nifit_scan.get_fdata()
        array = array.transpose(2, 0, 1)
        
        scan_name = scan.split(".")[0]
        for i, slice in enumerate(array):
            num = str(i).zfill(3)
            np.save(val_data_path + os.sep + scan_name  + f"_{num}.npy", slice)
        os.remove(val_data_path + scan)

    for scan in test_data:
        nifit_scan = nib.load(test_data_path + scan)
        array = nifit_scan.get_fdata()
        array = array.transpose(2, 0, 1)
        
        scan_name = scan.split(".")[0]
        for i, slice in enumerate(array):
            num = str(i).zfill(3)
            np.save(test_data_path + os.sep + scan_name  + f"_{num}.npy", slice)
        os.remove(test_data_path + scan)