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

class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 138,
            _Split.VAL: 138,
            _Split.TEST: 138,
        }
        return split_lengths[self]

class MC(VisionDataset):
    Split = _Split

    def __init__(
        self,
        split: "MC.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        
        self._root = root  
        self._image_folder_path = self._root + os.sep + "CXR_png"
        self._masks_path = self._root + os.sep + "ManualMask"
        self._split = split

        self.images = os.listdir(self._image_folder_path)
        self.class_id_mapping = {"background": 0, "left_lung": 1, "right_lung": 2}
        self.class_names = list(self.class_id_mapping.keys())
        self.masks = {"left_lung": os.listdir(self._masks_path + os.sep + "leftMask"),
                      "right_lung": os.listdir(self._masks_path + os.sep + "rightMask")}
        
        self._size_check()

    @property
    def split(self) -> "MC.Split":
        return self._split

    def _size_check(self):
        data_in_root = len(os.listdir(self._root))
        print(f"{self.split.length - data_in_root} scans are missing from {self._split.value.upper()} set")

    def get_length(self) -> int:
        return self.__len__()

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def get_image_data(self, index: int) -> np.ndarray:
        image_path = self._image_folder_path + os.sep + self.images[index]
        
        image = skimage.io.imread(image_path)
        image = np.stack((image,)*3, axis=0)
        image = torch.from_numpy(image).float()

        return image
    
    def get_target(self, index: int) -> np.ndarray:
        left_mask_path = self._masks_path + os.sep + "leftMask" + os.sep + self.masks["left_lung"][index]
        right_mask_path = self._masks_path + os.sep + "rightMask" + os.sep + self.masks["right_lung"][index]

        left_mask = skimage.io.imread(left_mask_path).astype(np.int_)
        right_mask = skimage.io.imread(right_mask_path).astype(np.int_)

        left_mask[left_mask==1] = self.class_id_mapping["left_lung"]
        right_mask[right_mask==1] = self.class_id_mapping["right_lung"]    

        target = left_mask + right_mask

        target = torch.from_numpy(target).unsqueeze(0)

        return target
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # Remove channel dim in target
        target = target.squeeze()

        return image, target