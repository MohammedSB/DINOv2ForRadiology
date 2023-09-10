import csv
from enum import Enum
import logging
import os
import shutil
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
            _Split.TRAIN: 76_522,
            _Split.VAL: 10_002,
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
        self.curr_imgs = os.listdir(root) #TODO: Remove

        # Set the labels dataframe
        labels_path = self._root + os.sep + "labels"
        self.labels = pd.read_csv(labels_path + ".csv")

        self._split = split
        self._define_split_dir() 
        self._extract_subset()
        self._check_size()

    def _check_size(self):
        num_of_images = len(os.listdir(self._split_dir))
        logger.info(f"{self.split.length - num_of_images} x-ray's are missing from {self._split.value.upper()} set")

    def _define_split_dir(self):
        self._split_dir = self._root + os.sep + self._split.value
        if self._split.value not in ["train", "val", "test"]:
            raise ValueError(f'Unsupported split "{self.split}"') 
        
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
        if self._split == _Split.TRAIN:
            subset = pd.read_csv(self._root + os.sep + "train_list.txt", names=["Image Index"])
        elif self._split == _Split.VAL:
            subset = pd.read_csv(self._root + os.sep + "val_list.txt", names=["Image Index"])
        elif self._split == _Split.TEST:
            subset = pd.read_csv(self._root + os.sep + "test_list.txt", names=["Image Index"])
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
        image_path = self._split_dir + os.sep + data_point["Image Index"]
        
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
    

def make_val_set(data_dir="/mnt/d/data/NIH"):
    train_val = pd.read_csv(data_dir + os.sep + "train_val_list.txt", names=["Image Index"])
    val_list = [i for i in range(len(train_val)-10_002, len(train_val))]
    val_set = train_val.iloc[val_list]
    train_set = train_val.drop(val_list)

    train_dir = data_dir + os.sep + "train"
    val_dir = data_dir + os.sep + "val"
    for image in val_set["Image Index"]:
        source = train_dir + os.sep + image
        dest = val_dir + os.sep + image
        shutil.move(source, dest)

    val_set.to_csv(data_dir + os.sep + "val_list.txt", index=False, header=False)
    train_set.to_csv(data_dir + os.sep + "train_list.txt", index=False, header=False)