from dataclasses import dataclass
from enum import Enum
from logging import CRITICAL
from typing import Tuple, Optional
from copy import deepcopy

import numpy as np
import pyvista as pv
#TODO: switch to pydantic dataclass, add typing support
#TODO: bump up python version to 3.9 and make appropriate changes to the type annotations

class SpatialUnit(Enum):
    PXL = "pixels"
    UM = "micrometers"
    M = "meters"

@dataclass
class SIM_3D_Data:
    """Dataclass to store 3D Structured Illumination Microscopy(SIM) data
    Note that the dimensions are in z, y, x order if not specified otherwise"""
    data: np.array 
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    unit: SpatialUnit = SpatialUnit.PXL
    name: str = None

    @property
    def shape(self):
        return self.data.shape

    def crop_data(self, x_range = None, y_range = None, z_range = None):
        cropped_SIM = deepcopy(self)
        slices = tuple(range_to_slice(r) for r in (z_range, y_range, x_range))
        cropped_SIM.data = cropped_SIM.data[slices]
        return cropped_SIM

    def apply(self, func):
        output_SIM = deepcopy(self)
        new_data = func(output_SIM.data)
        output_SIM.data = new_data
        return output_SIM

    def apply_along_axis(self, func, axis = 0):
        output_SIM = deepcopy(self)
        data = np.moveaxis(output_SIM.data, axis, 0)
        new_data = np.stack([func(plane) for plane in data], axis = 0)
        output_SIM.data = np.moveaxis(new_data, 0, axis)
        return output_SIM

    def to_pvUniformGrid(self) -> pv.UniformGrid:
        grid = pv.UniformGrid()
        grid.dimensions = np.array(self.shape)
        #grid.origin = np.array(origin)
        grid.spacing = np.array(self.scale)
        grid.point_data['values'] = self.data.flatten(order = 'F')
        #grid.field_data['name'] = [self.name]
        #grid.field_data['unit'] = [self.unit]
        return grid


def range_to_slice(range: Optional[Tuple[int, int]]) -> slice:
    if range is None:
        return slice(None)
    else:
        return slice(*range)


def cast_to_float32(SIM_image: SIM_3D_Data):
    input_data = SIM_image.data
    input_dtype = input_data.dtype

    if np.issubdtype(input_dtype, np.floating):
        SIM_image.data = np.float32(input_data)
    elif np.issubdtype(input_dtype, np.unsignedinteger):
        dtype_max_val = np.iinfo(input_dtype).max
        SIM_image.data = np.divide(input_data, dtype_max_val, dtype = np.float32)
    else:
        raise ValueError('Given SIM has invalid data type: that is, pixel values are neither floats nor unsigned integers')

    return SIM_image