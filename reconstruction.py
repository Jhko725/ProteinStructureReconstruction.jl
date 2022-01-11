#%%
%load_ext autoreload
%autoreload 2
%matplotlib qt

from typing import Union
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage.filters import threshold_local

from superresolution import SIM_3D_Data, SpatialUnit, cast_to_float32
from visualizeSIM import plot_overlay, plot_selection_box, plot_slices

#%%
def read_SIM(filepath: Union[str, Path]):
    #filepath = Path(filepath)
    file = h5py.File(filepath, 'r')
    data = file['Data']
    scale = tuple(data.attrs['pixel_sizes'])
    channels = data.attrs['channels']

    return [SIM_3D_Data(data[i, ...], scale, SpatialUnit.M, c) for i, c in enumerate(channels)]

#%%
SIM_data = read_SIM('./Fed_X63_Z3_SIM.h5')
SIM_data = [cast_to_float32(single_channel_data) for single_channel_data in SIM_data]

# %%
x_range, y_range = (2000, 2300), (1600, 1900)
cropped_SIM = [SIM.crop_data(x_range, y_range) for SIM in SIM_data]


fig, axes = plt.subplots(1, 2, figsize = (15, 8))
cmaps = ('Greens', 'Reds')

for ax, SIM, cmap in zip(axes, cropped_SIM, cmaps):
    ax = plot_overlay(SIM, ax, cmap = cmap)
    ax.set_title(f'{SIM.name.capitalize()} overlay')
    ax = plot_selection_box(x_range, y_range, ax, linewidth = 1, edgecolor = 'r', facecolor = 'none')
fig.suptitle(f'ROI for $x \in {x_range}$; $y \in {y_range}$')
fig.tight_layout()
# %%
channel = 1
plt.rc('font', size = 12)
fig, ax = plot_slices(cropped_SIM[channel], 5, 5, cmap = cmaps[channel])
fig.suptitle(f'{cropped_SIM[channel].name.capitalize()} channel, raw')
fig.tight_layout()
# %%
binarized_data = [SIM.apply_along_axis(lambda img: img > threshold_local(img, 23)) for SIM in cropped_SIM]
#%%
channel = 0
fig, ax = plot_slices(binarized_data[channel], 5, 5, cmap = cmaps[channel])
fig.suptitle(f'{binarized_data[channel].name.capitalize()} channel, binarized')
fig.tight_layout()
# %%
import pyvista as pv
pvgrid = binarized_data[0].to_pvUniformGrid()

plotter = pv.Plotter(notebook = False)
plotter.add_mesh(pvgrid)
plotter.show()
# %%
import pyvista as pv
plotter = pv.Plotter(notebook = False)
#print(plotter.ren_win.ReportCapabilities())
# %%
print(plotter.ren_win.ReportCapabilities())
# %%
pv.Report()
# %%
from pyvista import demos
demos.plot_wave()
# %%
