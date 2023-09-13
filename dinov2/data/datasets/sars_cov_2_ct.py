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
            _Split.TRAIN: 90,
            _Split.VAL: 50,
            _Split.TEST: 70,
        }
        return split_lengths[self]

class SARSCoV2CT(MedicalVisionDataset):
    Split = _Split

    def __init__(
        self,
        *,
        split: "SARSCoV2CT.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(split, root, transforms, transform, target_transform)
        
    @property
    def split(self) -> "SARSCoV2CT.Split":
        return self._split

    def get_length(self) -> int:
        return self.__len__()

    def get_num_classes(self) -> int:
        return 2
    
    def is_3d(self) -> bool:
        return True
    
    def is_multilabel(self) -> bool:
        return False

    def get_image_data(self, index: int) -> np.ndarray:
        scans_path = self._split_dir + os.sep + self.images[index]
        scans = os.listdir(scans_path)
        scans = [".".join(scan.split(".")[:-1]) for scan in scans]
        
        if scans[0].isnumeric():
            scans = [int(scan) for scan in scans]
            scans.sort()

        for i, scan in enumerate(scans):
            
            scan = skimage.io.imread(scans_path + os.sep + str(scan) + ".png")
            scan = scan[:, :, :3]
            scan = torch.from_numpy(scan).permute(2, 0, 1).float()

            scans[i] = scan 

        return scans
    
    def get_target(self, index: int) -> int:
        return int(int(self.images[index]) < 79)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int):
        images = self.get_image_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            for i in range(len(images)):
                images[i], target = self.transforms(images[i], target)
            images = torch.stack(images, dim=0)

        return images, target

def make_splits(data_dir="/mnt/z/data/SARS-CoV-2-CT"):
    i = 0

    covid_path = data_dir + os.sep + "Covid"
    covid_images = os.listdir(covid_path)

    healthy_path = data_dir + os.sep + "Healthy"
    healthy_images = os.listdir(healthy_path)

    other_path = data_dir + os.sep + "Others"
    other_images = os.listdir(other_path)

    all_images_path = data_dir + os.sep + "images" 
    os.makedirs(all_images_path, exist_ok=True)

    for img in covid_images:
        dest = covid_path + os.sep + f"{i}"
        os.rename(covid_path + os.sep + img, dest)
        shutil.move(dest, all_images_path)
        i+=1

    for img in healthy_images:
        dest = healthy_path + os.sep + f"{i}"
        os.rename(healthy_path + os.sep + img, dest)
        shutil.move(dest, all_images_path)
        i+=1

    for img in other_images:
        dest = other_path + os.sep + f"{i}"
        os.rename(other_path + os.sep + img, dest)
        shutil.move(dest, all_images_path)
        i+=1

    os.removedirs(covid_path)
    os.removedirs(healthy_path)
    os.removedirs(other_path)

    all_images = os.listdir(dir + os.sep + "images")
    all_images = [int(i) for i in all_images]
    all_images = np.sort(np.array(all_images))

    test_list = np.arange(0, 210, 210/70).round().astype("int")
    val_list = np.arange(0, 140, 140/50).round().astype("int")

    test_images = all_images[test_list]
    train_val_images = np.delete(all_images, test_list)

    val_images = train_val_images[val_list]
    train_images = np.delete(train_val_images, val_list)

    train_path = data_dir + os.sep + "train"
    val_path = data_dir + os.sep + "val" 
    test_path = data_dir + os.sep + "test"

    os.makedirs(train_path, exist_ok=True)
    for image in train_images:
        source = all_images_path + os.sep + str(image)
        shutil.move(source, train_path)

    os.makedirs(val_path, exist_ok=True)
    for image in val_images:
        source = all_images_path + os.sep + str(image)
        shutil.move(source, val_path)

    os.makedirs(test_path, exist_ok=True)
    for image in test_images:
        source = all_images_path + os.sep + str(image)
        shutil.move(source, test_path)

    os.removedirs(all_images_path)