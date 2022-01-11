from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.patches import Rectangle

from superresolution import SIM_3D_Data

def make_axis_if_none(ax: Optional[Axis]) -> Axis:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize = (10, 10))
    return ax
    

def plot_overlay(SIM_image: SIM_3D_Data, plt_axis: Optional[Axis] = None, projection_dim: int = 0, **imshow_kwargs) -> Axis:
    ax = make_axis_if_none(plt_axis)
    overlay = np.mean(SIM_image.data, axis = projection_dim)
    ax.imshow(overlay, **imshow_kwargs)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    fig = ax.get_figure()
    fig.tight_layout()
    return ax
     
def plot_selection_box(x_range, y_range, ax, **rectangle_kwargs):
    origin = (x_range[0], y_range[0])
    widths = (x_range[1]-x_range[0], y_range[1]-y_range[0])
    rect = Rectangle(origin, *widths, **rectangle_kwargs)
    ax.add_patch(rect)
    return ax

# TODO: add support for plotting slices along x and y axes as well.
# Will need to use transpose to swap that dimension with zero and proceed with the rest of the logic
def plot_slices(SIM_image: SIM_3D_Data, ncols: int, nrows: int, slice_dim: int = 0, **imshow_kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize = (10, 10))
    num_slices = SIM_image.shape[slice_dim]
    plot_inds = np.linspace(0, num_slices, ncols*nrows, endpoint = False)
    plot_inds = np.int_(np.floor(plot_inds))

    for i, ax in zip(plot_inds, axes.flat):
        ax.imshow(SIM_image.data[i], **imshow_kwargs)
        ax.set_title(f'Slice #{i}/{num_slices}')
    
    return fig, ax
