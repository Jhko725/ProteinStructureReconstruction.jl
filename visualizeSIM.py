from typing import Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from superresolution import SIM_3D_Data

def cmap_from_name(SIM_image):
    if SIM_image.name == 'desmin':
        cmap_name = 'Reds'
    else:
        cmap_name = 'Greens'
    cmap = mpl.cm.get_cmap(cmap_name).copy()
    cmap.set_under(color = 'white')
    return cmap


def make_axis_if_none(ax: Optional[Axis]) -> Axis:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize = (10, 10))
    return ax
    

def plot_overlay(SIM_image: SIM_3D_Data, plt_axis: Optional[Axis] = None, projection_dim: int = 0, **imshow_kwargs) -> Axis:
    ax = make_axis_if_none(plt_axis)
    overlay = np.mean(SIM_image.data, axis = projection_dim)
    img = ax.imshow(overlay, vmin = np.min(overlay[overlay>0]), **imshow_kwargs)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig = ax.get_figure()
    fig.colorbar(img, cax = cax, orientation = 'vertical', extend = 'both')
    fig.tight_layout()
    return ax


def plot_overlays(SIM_images: list[SIM_3D_Data], axes, cmap_factory):
    for i, (ax, SIM) in enumerate(zip(axes, SIM_images)):
        cmap = cmap_factory(i, SIM)
        ax = plot_overlay(SIM, ax, cmap = cmap, norm = mpl.colors.PowerNorm(gamma = 0.5))
        ax.set_title(f'{SIM.name.capitalize()}')
    return axes


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
