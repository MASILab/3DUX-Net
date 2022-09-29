from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *

from monai.transforms import (
    AsDiscreted,
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
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandRotate90d,
    EnsureTyped,
    Invertd,
    KeepLargestConnectedComponentd,
    SaveImaged,
    Activationsd
)

import numpy as np
from collections import OrderedDict
import glob

def data_loader(args):
    root_dir = args.root
    dataset = args.dataset

    print('Start to load data from directory: {}'.format(root_dir))

    if dataset == 'feta':
        out_classes = 8
    elif dataset == 'flare':
        out_classes = 5
    elif dataset == 'amos':
        out_classes = 16

    if args.mode == 'train':
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

        print('Finished loading all training samples from dataset: {}!'.format(dataset))
        print('Number of classes for segmentation: {}'.format(out_classes))

        return train_samples, valid_samples, out_classes

    elif args.mode == 'test':
        test_samples = {}

        ## Input inference data
        test_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTs', '*.nii.gz')))
        test_samples['images'] = test_img

        print('Finished loading all inference samples from dataset: {}!'.format(dataset))

        return test_samples, out_classes


def data_transforms(args):
    dataset = args.dataset
    if args.mode == 'train':
        crop_samples = args.crop_sample
    else:
        crop_samples = None

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

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
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

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    1.0, 1.0, 1.2), mode=("bilinear")),
                # ResizeWithPadOrCropd(keys=["image"], spatial_size=(168,168,128), mode=("constant")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
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

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(
                    1.5, 1.5, 2.0), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )


    if args.mode == 'train':
        print('Cropping {} sub-volumes for training!'.format(str(crop_samples)))
        print('Performed Data Augmentations for all samples!')
        return train_transforms, val_transforms

    elif args.mode == 'test':
        print('Performed transformations for all samples!')
        return test_transforms


def infer_post_transforms(args, test_transforms, out_classes):

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=test_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        ## If monai version <= 0.6.0:
        AsDiscreted(keys="pred", argmax=True, n_classes=out_classes),
        ## If moani version > 0.6.0:
        # AsDiscreted(keys="pred", argmax=True)
        # KeepLargestConnectedComponentd(keys='pred', applied_labels=[1, 3]),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.output,
                   output_postfix="seg", output_ext=".nii.gz", resample=True),
    ])

    return post_transforms



