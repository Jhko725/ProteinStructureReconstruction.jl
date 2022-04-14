#%%
%load_ext autoreload
%autoreload 2
%matplotlib tk

from typing import Union
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage.filters import threshold_local, threshold_otsu, threshold_sauvola, threshold_isodata, threshold_li

from superresolution import SIM_3D_Data, SpatialUnit, cast_to_float32
from visualizeSIM import *

#%%
def read_SIM(filepath: Union[str, Path]):
    #filepath = Path(filepath)
    file = h5py.File(filepath, 'r')
    data = file['Data']
    scale = tuple(data.attrs['pixel_sizes'])
    channels = data.attrs['channels']


    return [SIM_3D_Data(data[i, ...], scale, SpatialUnit.M, c) for i, c in enumerate(channels)]

#%%
SIM2_data2 = read_SIM('./Data/SIM2/Image2_SIM2.h5')
SIM2_data = read_SIM('./Data/SIM2/Image6_actin_SIM2.h5')
#SIM_data = [cast_to_float32(single_channel_data) for single_channel_data in SIM_data]
SIM2_data = [cast_to_float32(single_channel_data) for single_channel_data in SIM2_data]
SIM2_data2 = [cast_to_float32(single_channel_data) for single_channel_data in SIM2_data2]
#%%
fig, axes = plt.subplots(1, 2, figsize = (15, 8))
plt.rc('font', size = 15)
plt.rc('axes', labelsize = 15, titlesize = 15)
axes = plot_overlays(SIM2_data, axes, lambda i, SIM: cmap_from_name(SIM))
# %%
x_range, y_range = (800, 1300), (1300, 1800)
#x_range, y_range = (1000, 1500), (550, 1050)
x_range2, y_range2 = (1000, 1500), (250, 750)
#SIM_roi = [SIM.crop_data(x_range, y_range) for SIM in SIM_data]
SIM2_roi = [SIM.crop_data(x_range, y_range) for SIM in SIM2_data]

SIM2_roi2 = [SIM.crop_data(x_range2, y_range2) for SIM in SIM2_data2]
#%%
SIM2_data[0].shape



#for ax in axes:
#    ax = plot_selection_box(x_range, y_range, ax, linewidth = 1, edgecolor = 'k', facecolor = 'none')
fig.tight_layout()
fig.suptitle('SIM overlay images', fontsize = 'large')
#plt.savefig('./Figures/Image2SIM_powerlaw.png', bbox_inches = 'tight')
#%%
fig, axes = plt.subplots(1, 2, figsize = (15, 8))
plt.rc('font', size = 15)
plt.rc('axes', labelsize = 15, titlesize = 15)
axes = plot_overlays(SIM2_roi2, axes, lambda i, SIM: cmap_from_name(SIM))
#for ax in axes:
#    ax = plot_selection_box(x_range, y_range, ax, linewidth = 1, edgecolor = 'k', facecolor = 'none')
fig.tight_layout()
fig.suptitle(f'SIM$^2$ ROI for $x \in {x_range}$; $y \in {y_range}$', fontsize = 'large')
#%%
SIM_roi
#%%
for ax, SIM, cmap in zip(axes, cropped_SIM, cmaps):
    ax = plot_overlay(SIM, ax, cmap = cmap)
    ax.set_title(f'{SIM.name.capitalize()} overlay')
    ax = plot_selection_box(x_range, y_range, ax, linewidth = 1, edgecolor = 'r', facecolor = 'none')
fig.suptitle(f'ROI for $x \in {x_range}$; $y \in {y_range}$')
fig.tight_layout()

#%%
SIM2_data[0].scale
#%%
channel = 1
plt.rc('font', size = 12)
data = SIM2_roi[channel]
cmap = cmap_from_name(data)
fig, ax = plot_slices(data, 5, 5, cmap = cmap, norm = mpl.colors.PowerNorm(gamma = 0.5, vmin = 1e-6))
fig.suptitle(f'{data.name.capitalize()} channel, raw')
fig.tight_layout()
# %%
filter_size = 31
#binarized_data = [SIM.apply_along_axis(lambda img: img > threshold_local(img, filter_size)) for SIM in SIM2_roi]
binarized_data = [SIM.apply_along_axis(lambda img: img > threshold_li(img)) for SIM in SIM2_roi]
#%%
channel = 0
cmap = cmap_from_name((binarized_data[channel]))
fig, ax = plot_slices(binarized_data[channel], 5, 5, cmap = cmap)
fig.suptitle(f'{binarized_data[channel].name.capitalize()} channel,minimum cross-entropy binarization')
fig.tight_layout()
#%%
def cross_correlation(arr1, arr2):
    mean1, mean2 = np.mean(arr1), np.mean(arr2)
    std1, std2 = np.std(arr1), np.std(arr2)
    N1, N2 = len(arr1), len(arr2)
    return np.sum((arr1-mean1) * (arr2-mean2))/(std1*std2*N1*N2)

z_arr = np.arange(SIM2_roi[0].shape[0])
xcorr = np.array([cross_correlation(SIM2_roi[0].data[z], SIM2_roi[1].data[z]) for z in z_arr])
#%%
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
ax.plot(z_arr, xcorr_bin)
ax.set_xlabel('Z slice index (#)')
ax.set_ylabel('Normalized cross-correlation')
ax.grid(ls = '--')
ax.set_title('Cross-correlation between the two channels')
#%%
z_arr = np.arange(SIM2_roi[0].shape[0])
xcorr_bin = np.array([cross_correlation(binarized_data[0].data[z], binarized_data[1].data[z]) for z in z_arr])
#%%
import scipy.ndimage as ndimage

interp_SIM = [SIM[15:25, :, :].apply(lambda vol: ndimage.zoom(vol, (0.12593877/0.015651487, 1, 1))) for SIM in SIM2_roi2]
for SIM in interp_SIM:
    SIM.scale = (0.015651487, 0.015651487, 0.015651487)

interp_SIM[0].shape
#%%
bin_int_SIM = [SIM.apply_along_axis(lambda img: img > threshold_li(img)) for SIM in interp_SIM]
channel = 1
cmap = cmap_from_name((bin_int_SIM[channel]))
fig, ax = plot_slices(bin_int_SIM[channel], 5, 5, cmap = cmap, norm = mpl.colors.PowerNorm(gamma = 0.5, vmin = 1e-6))
fig.suptitle(f'{bin_int_SIM[channel].name.capitalize()} channel, Minimal cross-entropy binarization')
fig.tight_layout()
#%%
import pyvista as pv
channel = 0
cmap = ['kbc', 'fire'][channel]
pvgrid = bin_int_SIM[channel].to_pvUniformGrid()
pvgrid1 = bin_int_SIM[1].to_pvUniformGrid()
#%%
plotter = pv.Plotter(notebook = False)
plotter.add_mesh(pvgrid.threshold(value = 0.5).extract_geometry().smooth(n_iter = 500).elevation(), cmap = cmap, lighting = True, opacity = 1.0)
plotter.add_mesh(pvgrid1.threshold(value = 0.5).extract_geometry().smooth(n_iter = 500).elevation(), cmap = 'fire', lighting = True, opacity = 1.0)
plotter.camera_position = 'xy'
plotter.camera.elevation = 45
plotter.show_grid()
plotter.show()
#%%
data = bin_int_SIM[1].data
labels, n_labels = ndimage.label(data)
desmin_sizes = ndimage.labeled_comprehension(data, labels, np.arange(1, n_labels+1), len, float, 0)
#%%
data2 = bin_int_SIM[0].data
labels2, n_labels2 = ndimage.label(data2)
desmin_sizes2 = ndimage.labeled_comprehension(data2, labels2, np.arange(1, n_labels2+1), len, float, 0)
#%%
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
ax.hist([desmin_sizes*0.015651487**3, desmin_sizes2*0.015651487**3], bins = 20, label = ['Actin-Desmin', 'Desmin-Alpha-actinin'])
#ax.hist(desmin_sizes2, 50)
#ax.set_xscale('log', base = 10)
ax.set_yscale('log', base = 10)
ax.set_xlabel('Connected component volume ($\mu m^3$)')
ax.set_ylabel('Counts')
ax.grid(ls = '--')
ax.legend()
ax.set_title('Desmin connected component size distribution')
#%%
import scipy.stats as scstats
scstats.ks_2samp(desmin_sizes, desmin_sizes2)
#%%
#binarized_data = [SIM.apply_along_axis(lambda img: img > threshold_li(img)) for SIM in SIM2_roi]
interp_SIM[0]
#%%
from skimage.filters import sato, meijering, hessian, frangi

ridge_data = [SIM.apply_along_axis(lambda img: frangi(img, black_ridges = False, sigmas = range(1, 20, 4))) for SIM in interp_SIM]
#%%
ridge_data3d = [SIM.apply(lambda img: sato(img, black_ridges = False, sigmas = range(1, 15, 3))) for SIM in SIM2_roi]
#%%
ridge_data3d.shape
#%%
channel = 1
cmap = cmap_from_name(ridge_data[channel])
fig, ax = plot_slices(ridge_data[channel], 5, 5, cmap = cmap, norm = mpl.colors.PowerNorm(gamma = 0.5, vmin = 1e-6))
fig.suptitle(f'{ridge_data[channel].name.capitalize()} channel, binarized')
#%%
bin_ridge_data = [SIM.apply_along_axis(lambda img: img > threshold_li(img)) for SIM in ridge_data]
channel = 1
cmap = cmap_from_name(ridge_data[channel])
fig, ax = plot_slices(bin_ridge_data[channel], 5, 5, cmap = cmap, norm = mpl.colors.PowerNorm(gamma = 0.5, vmin = 1e-6))
fig.suptitle(f'{ridge_data[channel].name.capitalize()} channel, binarized')
#%%
import pyvista as pv
pvgrid = binarized_data[0].to_pvUniformGrid()

plotter = pv.Plotter(notebook = False)
plotter.add_mesh(pvgrid)
plotter.show()
# %%
import pyvista as pv
plotter = pv.Plotter(notebook = False)
plotter.show()
#print(plotter.ren_win.ReportCapabilities())
# %%
print(plotter.ren_win.ReportCapabilities())
# %%
pv.Report()
# %%
from pyvista import demos
demos.plot_wave()
# %%
channel = 0
cmap = ['fire', 'kbc'][channel]
pvgrid = bin_ridge_data[channel].to_pvUniformGrid()
pvgrid1 = bin_ridge_data[1].to_pvUniformGrid()
#%%
plotter = pv.Plotter(notebook = False)
plotter.add_mesh(pvgrid.threshold(value = 0.5).extract_geometry().smooth(n_iter = 500).elevation(), cmap = cmap, lighting = True, opacity = 1.0)
#plotter.add_mesh(pvgrid1.threshold(value = 0.5).extract_geometry().smooth(n_iter = 500).elevation(), cmap = 'kbc', lighting = True, opacity = 1.0)
plotter.camera_position = 'yx'
plotter.camera.elevation = 45
plotter.show_grid()
plotter.show()
# %%
