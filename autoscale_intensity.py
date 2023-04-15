#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from typing import Dict, Hashable, Mapping, Optional
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform
from monai.transforms.intensity.array import (
    ScaleIntensityRange,
)

class AutoScaleIntensity(MapTransform):
    """
    Derived from dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRange`.
    Auto-adjusts lower and upper limits of source array

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        b_min: intensity target range min.
        b_max: intensity target range max.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.b_min = b_min
        self.b_max = b_max
        self.dtype = dtype

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
    
        for key in self.key_iterator(d):
            a_min = np.min(d[key])
            a_max = np.max(d[key])
            # print(f'Autoscale [{a_min}, {a_max}] -> [{self.b_min}, {self.b_max}]')
            scaler = ScaleIntensityRange(a_min, a_max, self.b_min, self.b_max, True, self.dtype)
            d[key] = scaler(d[key])
        return d