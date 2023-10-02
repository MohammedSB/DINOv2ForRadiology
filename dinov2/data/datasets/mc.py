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
            _Split.TRAIN: 69,
            _Split.VAL: 23,
            _Split.TEST: 46,
        }
        return split_lengths[self]

class MC(MedicalVisionDataset):
    Split = _Split

    def __init__(
        self,
        split: "MC.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(split, root, transforms, transform, target_transform)
        
        self._masks_path = self._root + os.sep + "ManualMask"

        self.class_id_mapping = pd.DataFrame([0, 1, 2],
                                            index=["background", "left_lung", "right_lung"],
                                            columns=["class_id"])
        self.class_names = np.array(self.class_id_mapping.index)


    @property
    def split(self) -> "MC.Split":
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
        image = np.stack((image,)*3, axis=0)
        image = torch.from_numpy(image).float()

        return image
    
    def get_target(self, index: int) -> np.ndarray:

        img_name = self.images[index]

        left_mask_path = self._masks_path + os.sep + "leftMask" + os.sep + img_name
        right_mask_path = self._masks_path + os.sep + "rightMask" + os.sep + img_name

        left_mask = skimage.io.imread(left_mask_path).astype(np.int_)
        right_mask = skimage.io.imread(right_mask_path).astype(np.int_)

        left_mask[left_mask==1] = self.class_id_mapping.loc["left_lung"]["class_id"]
        right_mask[right_mask==1] = self.class_id_mapping.loc["right_lung"]["class_id"]  

        target = left_mask + right_mask

        target = torch.from_numpy(target).unsqueeze(0)

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
    
def make_splits(data_dir="/mnt/d/data/MC"):
    image_path = data_dir + os.sep + "CXR_png"

    # Define the indices for val and test
    test_list = [i for i in range(0, 138, math.ceil(138/46))]
    val_list = [i for i in range(0, 92, math.ceil(92/23))]
    entire_data = pd.DataFrame(os.listdir(image_path))

    test_set = entire_data.iloc[test_list]
    train_val_set = entire_data.drop(test_list).reset_index(drop=True)

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

    os.rmdir(image_path)