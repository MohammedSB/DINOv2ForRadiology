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
            _Split.TRAIN: 86_524,
            _Split.VAL: 86_524,
            _Split.TEST: 25_596,
        }
        return split_lengths[self]

class NIHChestXray(VisionDataset):
    Split = _Split
    MULTILABEL = True

    def __init__(
        self,
        *,
        split: "NIHChestXray.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        
        # Define paths for the data
        self._root = root 
        self._data_directory = ("/").join(root.split("/")[:-1]) # This defines the root for the entire data directory  
        self.curr_imgs = os.listdir(root) #TODO: Remove

        # Set the labels dataframe
        labels_path = self._data_directory + os.sep + "labels"
        self.labels = pd.read_csv(labels_path + ".csv")

        self._split = split
        self._extract_subset()
        self._size_check()

    def _size_check(self):
        data_in_root = len(os.listdir(self._root))
        logger.info(f"{self.split.length - data_in_root} x-ray's are missing from {self._split.value.upper()} set")

    def _clean_labels(self):
        # Define inner split string function
        def spilt_string(string):
            splitted = string.split("|")
            return splitted

        # Turn all labels into list
        self.labels["Finding Labels"] = self.labels["Finding Labels"].apply(spilt_string)

        # Encoding of multilabeled targets
        mlb = preprocessing.MultiLabelBinarizer()
        targets = mlb.fit_transform(self.labels["Finding Labels"])
        self.class_names = mlb.classes_
        self._class_ids = [i for i in range(1, len(self.class_names)+1)]
        self.targets = pd.DataFrame(targets, columns=mlb.classes_).to_numpy()

    def _extract_subset(self):
        # Define either train or testset
        if self._split == _Split.TRAIN or self._split == _Split.VAL:
            subset = pd.read_csv(self._data_directory + os.sep + "train_val_list.txt", names=["Image Index"])
        elif self._split == _Split.TEST:
            subset = pd.read_csv(self._data_directory + os.sep + "test_list.txt", names=["Image Index"])
        else:
            raise ValueError(f'Unsupported split "{self.split}"')

        self.labels = pd.merge(self.labels, subset, how="inner", on=["Image Index"])

        #TODO : remove
        # to_add = []
        # for i in self.labels.index:
        #     if self.labels.iloc[i]["Image Index"] in self.curr_imgs:
        #         to_add.append(i)

        # self.labels = self.labels.iloc[to_add]
        self._clean_labels()

    @property
    def split(self) -> "NIHChestXray.Split":
        return self._split
    
    def _get_class_ids(self) -> list:
        return self._class_ids
    
    def _get_class_names(self) -> list:
        return self.class_names
    
    def get_length(self) -> int:
        return self.__len__()
    
    def get_num_classes(self) -> int:
        return len(self.class_names)

    def find_class_id(self, class_index: int) -> str:
        class_ids = self._get_class_ids()
        return str(class_ids[class_index])

    def find_class_name(self, class_index: int) -> str:
        class_names = self._get_class_names()
        return str(class_names[class_index])

    def get_image_data(self, index: int) :
        data_point = self.labels.iloc[index]
        image_path = self._root + os.sep + data_point["Image Index"]
        
        # Read as gray because some of the images have extra layers in the 3rd dimension
        image = skimage.io.imread(image_path, as_gray=True).astype(np.float16)
        image = np.stack((image,)*3, axis=0)
        image = torch.from_numpy(image)

        return image

    def get_target(self, index: int):
        return self.targets[index]

    def get_targets(self) -> np.ndarray:
        return self.targets

    def get_class_id(self, index: int) -> str:
        class_id = self.targets[index]
        return str(class_id)

    def get_class_name(self, index: int) -> str:
        class_name_index = self.targets[index]
        class_name = self.class_names[class_name_index]
        return str(class_name)
    
    def __getitem__(self, index):
        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.labels)