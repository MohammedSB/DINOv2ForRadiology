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

logging.getLogger('nibabel').setLevel(logging.CRITICAL)

class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 15,
            _Split.VAL: 15,
            _Split.TEST: 20,
        }
        return split_lengths[self]

class BTCV(MedicalVisionDataset):
    Split = _Split

    def __init__(
        self,
        *,
        split: "BTCV.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(split, root, transforms, transform, target_transform)

        self._image_path = self._split_dir + os.sep + "img"
        self.images = os.listdir(self._image_path)

        self.labels = None
        if self._split != _Split.TEST:
            self._labels_path = self._split_dir + os.sep + "label"
            self.labels = os.listdir(self._labels_path)
    
        self.class_id_mapping = {    
            0: "background",
            1: "spleen",
            2: "rkid",
            3: "lkid",
            4: "gall",
            5: "eso",
            6: "liver",
            7: "sto",
            8: "aorta",
            9: "IVC",
            10: "veins",
            11: "pancreas",
            12: "rad",
            13: "lad"
        }
        self.class_names = list(self.class_id_mapping.keys())

    def _check_size(self):
        num_of_images = len(os.listdir(self._split_dir + os.sep + "img"))
        print(f"{self._split.length - num_of_images} scans are missing from {self._split.value.upper()} set")

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def is_3d(self) -> bool:
        return True

    def get_image_data(self, index: int, return_affine_matrix=False) -> np.ndarray:
        image_folder_path = self._image_path + os.sep + self.images[index]
        image_path = image_folder_path + os.sep + os.listdir(image_folder_path)[0]  

        nifti_image = nib.load(image_path)
        image = nifti_image.get_fdata()
        image = np.stack((image,)*3, axis=0)
        image = torch.from_numpy(image).permute(3, 0, 1, 2).float()

        if return_affine_matrix:
            affine = nifti_image.affine
            return image, affine
        return image
    
    def get_target(self, index: int) -> Tuple[np.ndarray, torch.Tensor, None]:
        if self.split == _Split.TEST:
            return None

        label_folder_path = self._labels_path + os.sep + self.labels[index]
        label_path = label_folder_path + os.sep + os.listdir(label_folder_path)[0]  
        
        target = nib.load(label_path).get_fdata()
        target = torch.from_numpy(target).unsqueeze(0)
        target = target.permute(3, 0, 1, 2)
    
        return target
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):

        image = self.get_image_data(index)
        target = self.get_target(index)

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        if self.transform is not None:
            transformed_image = []
            for i in range(len(image)):
                np.random.seed(seed), torch.manual_seed(seed) 
                transformed_image.append(self.transform(image[i]))
            image = torch.stack(transformed_image, dim=0)

        if self.target_transform is not None and target is not None:
            transformed_target = []
            for i in range(len(target)):
                np.random.seed(seed), torch.manual_seed(seed) 
                transformed_target.append(self.target_transform(target[i]))
            target = torch.stack(transformed_target, dim=0).squeeze()

        return image, target

def make_splits(data_dir = "/mnt/z/data/Abdomen/RawData"):
    train_path = data_dir + os.sep + "train"
    test_path = data_dir + os.sep + "test"
    if "Training" in os.listdir(data_dir): 
        os.rename(data_dir + os.sep + "Training", train_path)
    if "Testing" in os.listdir(data_dir): 
        os.rename(data_dir + os.sep + "Testing", test_path)


    train_image_path = train_path + os.sep + "img"
    train_label_path = train_path + os.sep + "label"
    train_images, train_labels = os.listdir(train_image_path), os.listdir(train_label_path)

    val_path = data_dir + os.sep + "val"
    val_image_path = val_path + os.sep + "img"
    val_label_path = val_path + os.sep + "label"
    os.makedirs(val_path, exist_ok=True), os.makedirs(val_image_path, exist_ok=True), os.makedirs(val_label_path, exist_ok=True)

    val_image_set = [img for i, img in enumerate(train_images) if i % 2 == 0]
    for img in train_images:
        if img in val_image_set:
            shutil.move(train_image_path + os.sep + img, val_image_path)

    val_label_set = [img for i, img in enumerate(train_labels) if i % 2 == 0]
    for label in train_labels:
        if label in val_label_set:
            shutil.move(train_label_path + os.sep + label, val_label_path)