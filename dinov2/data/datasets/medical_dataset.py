import logging
import os
from typing import Callable, Optional
from torchvision.datasets import VisionDataset
from abc import ABC, abstractmethod

logger = logging.getLogger("dinov2")

class MedicalVisionDataset(VisionDataset):
    def __init__(
        self,
        split, 
        root: str,         
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self._root = root
        self._split = split

        self._define_split_dir()
        self._check_size()
        
        self.images = os.listdir(self._split_dir)

    def _check_size(self):
        num_of_images = len(os.listdir(self._split_dir))
        logger.info(f"{self._split.length - num_of_images} scans are missing from {self._split.value.upper()} set")

    def _define_split_dir(self):
        self._split_dir = self._root + os.sep + self._split.value
        if self._split.value not in ["train", "val", "test"]:
            raise ValueError(f'Unsupported split "{self.split}"')
         
    @abstractmethod
    def is_3d(self) -> bool:
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        pass