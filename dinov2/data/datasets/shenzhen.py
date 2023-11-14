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

logger = logging.getLogger("dinov2")
_Target = int

class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 361,
            _Split.VAL: 91,
            _Split.TEST: 114,
        }
        return split_lengths[self]

class Shenzhen(MedicalVisionDataset):
    Split = _Split

    def __init__(
        self,
        *,
        split: "Shenzhen.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(split, root, transforms, transform, target_transform)

        self._masks_path = self._root + os.sep + "masks"

        self.class_id_mapping = pd.DataFrame([0, 1],
                                            index=["background", "lung"],
                                            columns=["class_id"])
        self.class_names = np.array(self.class_id_mapping.index)

    @property
    def split(self) -> "Shenzhen.Split":
        return self._split

    def get_length(self) -> int:
        return self.__len__()

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def is_3d(self) -> bool:
        return False

    def get_image_data(self, index: int) -> np.ndarray:
        image_path = self._split_dir + os.sep + self.images[index]
        
        image = skimage.io.imread(image_path)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image
    
    def get_target(self, index: int) -> np.ndarray:

        mask_path = self.images[index].replace(".png", "_mask.png") 
        mask_path = self._masks_path + os.sep + mask_path
        mask = skimage.io.imread(mask_path).astype(np.int_)

        mask[mask==255] = self.class_id_mapping.loc["lung"]["class_id"] 

        target = torch.from_numpy(mask).unsqueeze(0)

        return target
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.get_image_data(index)
        target = self.get_target(index)

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        if self.transform is not None:
            np.random.seed(seed), torch.manual_seed(seed) 
            image = self.transform(image)

        if self.target_transform is not None:
            np.random.seed(seed), torch.manual_seed(seed) 
            target = self.target_transform(target)

        # Remove channel dim in target
        target = target.squeeze()

        return image, target

def make_splits(data_dir="/mnt/d/data/Shenzhen"):
    image_path = data_dir + os.sep + "CXR_png"
    mask_path = data_dir + os.sep + "masks"
    
    image_names = ["_".join(image_name.split(".")[:-1]) for image_name in os.listdir(image_path)] 
    mask_names = ["_".join(mask_name.split("_")[:-1]) for mask_name in os.listdir(mask_path)]

    # all images with masks
    images = pd.DataFrame([image+".png" for image in image_names if image in mask_names])

    # define the indices for val and test
    test_list = [i for i in range(0, 566, math.ceil(566/114))]
    val_list = [i for i in range(0, 452, math.ceil(452/91))]

    test_set = images.iloc[test_list]
    train_val_set = images.drop(test_list).reset_index(drop=True)

    val_set = train_val_set.iloc[val_list]
    train_set = train_val_set.drop(val_list)

    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(data_dir + os.sep + split, exist_ok=True)

    # add train images to train folder
    train_dir = data_dir + os.sep + "train"
    for image in train_set[0]:
        source = image_path + os.sep + image
        dest = train_dir + os.sep + image
        shutil.move(source, dest)

    # add validation images
    val_dir = data_dir + os.sep + "val"
    for image in val_set[0]:
        source = image_path + os.sep + image
        dest = val_dir + os.sep + image
        shutil.move(source, dest)

    # add test images
    test_dir = data_dir + os.sep + "test"
    for image in test_set[0]:
        source = image_path + os.sep + image
        dest = test_dir + os.sep + image
        shutil.move(source, dest)
