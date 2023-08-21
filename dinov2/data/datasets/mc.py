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
        
        self.masks = {"left_lung": os.listdir(self._masks_path + os.sep + "leftMask"),
                      "right_right": os.listdir(self._masks_path + os.sep + "rightMask")}

    @property
    def split(self) -> "MC.Split":
        return self._split

    def _size_check(self):
        data_in_root = len(os.listdir(self._root))
        print(f"{self.split.length - data_in_root} scans are missing from {self._split.value.upper()} set")

    def get_image_data(self, index: int) -> np.ndarray:
        image_path = self._image_folder_path + os.sep + self.images[index]
        
        image = skimage.io.imread(image_path)
        image = np.stack((image,)*3, axis=0)
        image = torch.from_numpy(image).float()

        return image
    
    def get_target(self, index: int) -> dict:
        left_mask_path = self._masks_path + os.sep + "leftMask" + os.sep + self.masks["left_lung"][index]
        right_mask_path = self._masks_path + os.sep + "leftMask" + os.sep + self.masks["right_right"][index]

        left_mask = skimage.io.imread(left_mask_path).astype(np.uint8)
        right_mask = skimage.io.imread(right_mask_path).astype(np.uint8)

        masks = {"left_lung": left_mask,
                "right_lung": right_mask}
        return masks

    def get_targets(self) -> dict:
        return self.masks 
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.get_image_data(index)
        masks = self.get_target(index)

        if self.transforms is not None:
            image, masks = self.transforms(image, masks)

        return image, masks