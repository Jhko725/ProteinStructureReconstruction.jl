from typing import Tuple, Optional, List, Dict
import copy

import pyvista as pv
import numpy as np
import pandas as pd

Bounds = Optional[Tuple[float, float]]

class PointCloud3D():

    def __init__(self, coordinates):
        #assert coordinates.shape[1] == 3, "Coordinates must be a numpy array of 3D coordinates with shape (N, 3)!"
        self.pointcloud = pv.PolyData(coordinates)

    def __repr__(self):
        return self.pointcloud.__repr__()

    def __len__(self):
        return self.pointcloud.n_points

    @property
    def coordinates(self):
        return self.pointcloud.points

    @property
    def shape(self):
        return self.coordinates.shape
    
    @property
    def bounds(self):
        bounds_concat = tuple(self.pointcloud.bounds)
        bounds_x, bounds_y, bounds_z = bounds_concat[0:2], bounds_concat[2:4], bounds_concat[4:6]
        return bounds_x, bounds_y, bounds_z

    def add_field(self, field_name: str, field_data: np.ndarray):
        assert field_data.shape[0] == self.shape[0], "Field data must be a 2D numpy array of shape (N, F), where N is the number of coordinates and F is the feature dimension!"
        self.pointcloud[field_name] = field_data

    def clip_with_bounds(self, bounds_x: Bounds = None, bounds_y: Bounds = None, bounds_z: Bounds = None):
        
        clipped_PointCloud3D = copy.deepcopy(self)
        if bounds_x == bounds_y == bounds_z == None: return clipped_PointCloud3D
        
        bounds_xyz = self._handle_none_bounds(bounds_x, bounds_y, bounds_z)
        bounds_concat = sum(bounds_xyz, ())
        clipped_PointCloud3D.pointcloud = clipped_PointCloud3D.pointcloud.clip_box(bounds_concat, invert = False)
        return clipped_PointCloud3D

    def _handle_none_bounds(self, bounds_x: Bounds, bounds_y: Bounds, bounds_z: Bounds):
        new_bounds = (bounds_x, bounds_y, bounds_z)
        current_bounds = self.bounds

        return tuple(new_bnd if new_bnd != None else cur_bnd for new_bnd, cur_bnd in zip(new_bounds, current_bounds))

    @classmethod
    def from_DataFrame(cls, dataframe: pd.DataFrame, coordinate_labels: List, field_labels_map: Dict):
        new_PointCloud = cls(dataframe[coordinate_labels].to_numpy())
        for label_df, label_new in field_labels_map.items():
            new_PointCloud.add_field(str(label_new), dataframe[label_df].to_numpy())

        return new_PointCloud

    def make_uniform_analogue(self, random_seed = 0):
        '''
        Returns a uniform randomly distributed point cloud with the same spatial domain and number of points as the given point cloud
        '''
        rng = np.random.default_rng(random_seed)
        low, high = tuple(zip(*self.bounds))
        points_array = rng.uniform(low, high, size = self.shape)
        return PointCloud3D(points_array)