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
        self.images = np.sort(np.array(os.listdir(self._image_path)))

        self.labels = None
        if self._split != _Split.TEST:
            self._labels_path = self._split_dir + os.sep + "label"
            self.labels = np.sort(np.array(os.listdir(self._labels_path)))
    
        self.class_id_mapping = pd.DataFrame([i for i in range(14)],
                                    index=["background", "spleen", "rkid", "lkid", "gall", "eso",
                                           "liver", "sto", "aorta", "IVC", "veins", "pancreas",
                                           "rad", "lad"],
                                    columns=["class_id"])
        self.class_names = np.array(self.class_id_mapping.index)

    def _check_size(self):
        num_of_images = len(os.listdir(self._split_dir + os.sep + "img"))
        logging.info(f"{self._split.length - num_of_images} scans are missing from {self._split.value.upper()} set")

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def is_3d(self) -> bool:
        return True

    def get_image_data(self, index: int, seed: int = 0, return_affine_matrix=False) -> np.ndarray:
        image_folder_path = self._image_path + os.sep + self.images[index]
        image_path = image_folder_path + os.sep + os.listdir(image_folder_path)[0]  

        if self._split == _Split.TRAIN:
            nifti_image = nib.load(image_path, mmap=False)
            proxy = nifti_image.dataobj
            slice_indices = proxy.shape[-1] - 1
            np.random.seed(seed)
            start = np.random.randint(0, slice_indices-10)
            indices = list(range(start, start+10))
            image = np.array([proxy[..., i] for i in indices])
        else:
            nifti_image = nib.load(image_path)
            image = nifti_image.get_fdata()
            image = image.transpose(2, 0, 1)

        image = np.stack((image,)*3, axis=0)
        image = torch.from_numpy(image).permute(1, 0, 2, 3).float()

        if return_affine_matrix:
            affine = nifti_image.affine
            return image, affine
        
        # pre-preprocess
        image = torch.clamp(image, max=600)
        return image
    
    def get_target(self, index: int, seed: int = 0) -> Tuple[np.ndarray, torch.Tensor, None]:
        if self.split == _Split.TEST:
            return None

        label_folder_path = self._labels_path + os.sep + self.labels[index]
        label_path = label_folder_path + os.sep + os.listdir(label_folder_path)[0]  

        if self._split == _Split.TRAIN:
            nifti_image = nib.load(label_path, mmap=False)
            proxy = nifti_image.dataobj
            slice_indices = proxy.shape[-1] - 1
            np.random.seed(seed)
            start = np.random.randint(0, slice_indices-10)
            indices = list(range(start, start+10))
            target = np.array([proxy[..., i] for i in indices])
        else:
            target = nib.load(label_path).get_fdata()
            target = target.transpose(2, 0, 1)
        
        target = torch.from_numpy(target).unsqueeze(0).long()
        target = target.permute(1, 0, 2, 3)
    
        return target
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):

        seed = np.random.randint(2147483647) # make a seed with numpy generator 

        image = self.get_image_data(index, seed=seed)
        target = self.get_target(index, seed=seed)

        if self.transform is not None:
            transformed_image = []
            for i in range(len(image)):
                np.random.seed(seed), torch.manual_seed(seed) 
                transformed_image.append(self.transform(image[i]))
            image = torch.stack(transformed_image, dim=0).squeeze()

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