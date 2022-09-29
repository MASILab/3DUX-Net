from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    ToTensor,
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandRotate90d
)

import numpy as np
from collections import OrderedDict
import glob

def data_loader(root_dir, dataset):
    train_samples = {}
    valid_samples = {}

    ## Input training data
    train_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTr', '*.nii.gz')))
    train_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTr', '*.nii.gz')))
    train_samples['images'] = train_img
    train_samples['labels'] = train_label

    ## Input validation data
    valid_img = sorted(glob.glob(os.path.join(root_dir, 'imagesVal', '*.nii.gz')))
    valid_label = sorted(glob.glob(os.path.join(root_dir, 'labelsVal', '*.nii.gz')))
    valid_samples['images'] = valid_img
    valid_samples['labels'] = valid_label

    if dataset == 'feta':
        out_classes = 8
    elif dataset == 'flare':
        out_classes = 5
    elif dataset == 'amos':
        out_classes = 16

    return train_samples, valid_samples, out_classes

def data_transforms(dataset, crop_samples):
    if dataset == 'feta':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

    elif dataset == 'flare':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(256,256,128), mode=("constant")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

    elif dataset == 'amos':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

    return train_transforms, val_transforms



