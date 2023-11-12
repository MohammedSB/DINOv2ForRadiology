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
            _Split.TRAIN: 2013,
            _Split.VAL: 1766,
        }
        return split_lengths[self]

class BTCVSlice(MedicalVisionDataset):
    Split = _Split

    def __init__(
        self,
        *,
        split: "BTCVSlice.Split",
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
        return False

    def get_image_data(self, index: int) -> np.ndarray:
        image_path = self._image_path + os.sep + self.images[index]
        image = np.load(image_path)
        image = np.stack((image,)*3, axis=0)
        image = torch.tensor(image).float()

        # pre-preprocess
        image = torch.clamp(image, min=-1024, max=600)
        return image
    
    def get_target(self, index: int) -> Tuple[np.ndarray, torch.Tensor, None]:
        if self.split == _Split.TEST:
            return None
        
        label_path = self._labels_path + os.sep + self.labels[index]
        label = np.load(label_path)
        label = torch.from_numpy(label).unsqueeze(0)

        return label
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):

        seed = np.random.randint(2147483647) # make a seed with numpy generator 

        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transform is not None:
            np.random.seed(seed), torch.manual_seed(seed) 
            image = self.transform(image)

        if self.target_transform is not None and target is not None:
            np.random.seed(seed), torch.manual_seed(seed) 
            target = self.target_transform(target)

        # Remove channel dim in target
        target = target.squeeze()

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

def slice_it(path="/mnt/z/data/BTCVSlice"):
    train_path = path + "/train/img/"
    path_lbl = train_path.replace("img", "label")
    images = os.listdir(train_path)
    for image_folder in images:
        label_folder = image_folder.replace('img', 'label')

        image_name = os.listdir(train_path + f"/{image_folder}")
        label_name = os.listdir(path_lbl + f"/{label_folder}")

        image_folder_path = train_path + os.sep + image_folder
        image_path = image_folder_path + os.sep + image_name[0]

        label_folder_path = path_lbl + os.sep + label_folder
        label_path = label_folder_path + os.sep + label_name[0]

        nifti_image = nib.load(image_path)
        image = nifti_image.get_fdata()
        image = image.transpose(2, 0, 1)

        nifti_label = nib.load(label_path)
        label = nifti_label.get_fdata()
        label = label.transpose(2, 0, 1)

        for i, slice in enumerate(image):
            num = str(i).zfill(3)
            np.save(train_path + os.sep + image_folder.split(".")[0] + f"_{num}.npy", slice)
            np.save(path_lbl + os.sep + label_folder.split(".")[0] + f"_{num}.npy", label[i])

    val_path = path + "/val/img/"
    path_lbl = val_path.replace("img", "label")
    images = os.listdir(val_path)
    for image_folder in images:
        label_folder = image_folder.replace('img', 'label')

        image_name = os.listdir(val_path + f"/{image_folder}")
        label_name = os.listdir(path_lbl + f"/{label_folder}")

        image_folder_path = val_path + os.sep + image_folder
        image_path = image_folder_path + os.sep + image_name[0]

        label_folder_path = path_lbl + os.sep + label_folder
        label_path = label_folder_path + os.sep + label_name[0]

        nifti_image = nib.load(image_path)
        image = nifti_image.get_fdata()
        image = image.transpose(2, 0, 1)

        nifti_label = nib.load(label_path)
        label = nifti_label.get_fdata()
        label = label.transpose(2, 0, 1)

        for i, slice in enumerate(image):
            num = str(i).zfill(3)
            np.save(val_path + os.sep + image_folder.split(".")[0] + f"_{num}.npy", slice)
            np.save(path_lbl + os.sep + label_folder.split(".")[0] + f"_{num}.npy", label[i])