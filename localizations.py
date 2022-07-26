#%%
%load_ext autoreload
%autoreload 2
import numpy as np
import pyvista as pv
from more_itertools import always_iterable
import colorcet

from fileio import *
from pointcloud import *


STORM_data_dir = './Data/STORM'
#exp_name = 'desmin_alphaactinin_600nm'
exp_name = 'desmin_alphaactinin_600nm'
cmaps = ['kbc', 'fire']
exp_filepath = fetch_STORM_experiment_filepath(exp_name, STORM_data_dir)

localization_df = drop_invalid_localizations(read_Vutara352_localization_data(exp_filepath))
probe_ids, localization_dfs = get_per_probe_localizations(localization_df)
#%%
localization_df.head()
#%%
localization_df.to_csv(f'./Data/STORM/{exp_name}.csv', index = False)
# %%

point_clouds = [PointCloud3D.from_DataFrame(df, coordinate_labels = ['x', 'y', 'z'], field_labels_map = {'amp': 'Intensity', 'z': 'Z'}) for df in localization_dfs]

#%%
print(point_clouds[1])
# %%
def plot_point_clouds(point_clouds, colormaps, plotter = None, **add_mesh_kwargs):
    if plotter == None: plotter = pv.Plotter()
    point_clouds, colormaps = always_iterable(point_clouds), always_iterable(colormaps)
    actors = [plotter.add_mesh(pt_cld.pointcloud, cmap = cmap, **add_mesh_kwargs) for pt_cld, cmap in zip(point_clouds, colormaps)]
    plotter.show_grid(xlabel = 'X (nm)', ylabel = 'Y (nm)', zlabel = 'Z (nm)')

    return plotter, actors
#%%
plotter = pv.Plotter(off_screen = False)
plot_kwargs = {'point_size': 1.0, 'opacity': 1.0, 'lighting': True, 'clim': (0.0, 150.0), 'ambient': 0.9, 'show_scalar_bar': False}
actor1 = plotter.add_mesh(point_clouds[0].pointcloud.project_points_to_plane(), cmap = 'kr', **plot_kwargs)
actor2 = plotter.add_mesh(point_clouds[1].pointcloud.project_points_to_plane(), cmap = 'kg', **plot_kwargs)
plotter.show_grid(xlabel = 'X (nm)', ylabel = 'Y (nm)', zlabel = 'Z (nm)')
plotter.camera_position = 'xy'
plotter.show()
#plotter.screenshot(f'./Figures/{exp_name}_overlay_desmin.png')
# %%
add_mesh_kwargs = {'point_size': 1.0, 'opacity': 0.5, 'lighting': True, 'ambient': 0.5, 'show_scalar_bar': False}
plotter, _ = plot_point_clouds(point_clouds, cmaps, scalars = 'Z', **add_mesh_kwargs)
#plotter.camera_position = 'xy'
plotter.show()
#plotter.screenshot(f'./Figures/{exp_name}.png')

# %%

xbounds, ybounds = (6000, 11000), (2000, 7000)
clipped_ptclouds = [ptcloud.clip_with_bounds(xbounds, ybounds) for ptcloud in point_clouds]

add_mesh_kwargs = {'opacity': 0.8, 'ambient': 0.5, 'show_scalar_bar': True}

plotter = pv.Plotter(off_screen = False)
plotter.enable_eye_dome_lighting()

plotter, _ = plot_point_clouds(clipped_ptclouds[0], cmaps[0], plotter = plotter, point_size = 1.5, **add_mesh_kwargs)
plotter, _ = plot_point_clouds(clipped_ptclouds[1], cmaps[1], plotter = plotter, point_size = 3.0, **add_mesh_kwargs)
#plotter.enable_eye_dome_lighting()
#plotter.camera_position = 'xy'
plotter.show()

#%%
import hdbscan
from joblib import Memory
from tqdm import tqdm
idx = 1
#%%
cluster_sizes = np.arange(5, 120, 3, dtype = int)
n_clusters = np.zeros_like(cluster_sizes)
n_noise = np.zeros_like(cluster_sizes)
for i, c in tqdm(enumerate(cluster_sizes)):
    clusterer = hdbscan.HDBSCAN(min_cluster_size = int(c), min_samples = 1, cluster_selection_epsilon = 35, cluster_selection_method = 'eom', memory=Memory(cachedir=None))
    clusterer.fit(clipped_ptclouds[idx].coordinates)
    n_clusters[i] = np.max(clusterer.labels_)+1
    n_noise[i] = np.sum(clusterer.labels_ == -1)

noise_ratio = n_noise/len(clipped_ptclouds[idx])
#%%
%matplotlib inline
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
ax.plot(cluster_sizes, n_clusters)
ax2 = ax.twinx()
ax2.plot(cluster_sizes, noise_ratio)
#%%
n_clusters[1]
#%%
def make_polydata(coordinates):
    polydata = pv.PolyData(coordinates)
    polydata["Z"] = coordinates[:,-1]
    return PointCloud3D(polydata)

def cluster_point_cloud(point_cloud, **HDBSCAN_params):
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_params)
    clusterer.fit(point_cloud.coordinates)

    noise_inds = clusterer.labels_ == -1

    intensity = point_cloud.pointcloud["Intensity"]
    foreground_pts = point_cloud.coordinates[np.bitwise_not(noise_inds)]
    noise_pts = point_cloud.coordinates[noise_inds]

    noise_ptcld = make_polydata(noise_pts)
    noise_ptcld.add_field('Intensity', intensity[noise_inds])
    signal_ptcld = make_polydata(foreground_pts)
    signal_ptcld.add_field('Cluster ID', clusterer.labels_[np.bitwise_not(noise_inds)])
    signal_ptcld.add_field('Intensity', intensity[np.bitwise_not(noise_inds)])

    return signal_ptcld, noise_ptcld
#%%
hdbscan_params = {'min_cluster_size': 20, 'min_samples': 1, 'cluster_selection_method': 'eom', 'cluster_selection_epsilon': 35}
signal_ptcld, noise_ptcld = cluster_point_cloud(clipped_ptclouds[idx], **hdbscan_params)

# %%

#%%
plotter = pv.Plotter(off_screen = False)
plotter.enable_eye_dome_lighting()

add_mesh_kwargs = {'point_size': 2.0, 'opacity': 0.8, 'ambient': 0.5, 'show_scalar_bar': True}
plotter, _ = plot_point_clouds([noise_ptcld, signal_ptcld], ['gray', cmaps[idx]], scalars = 'Z', **add_mesh_kwargs)
#plotter.screenshot(f'./Figures/{exp_name}.png')
plotter.show()
# %%
plotter = pv.Plotter(off_screen = False)
plotter.enable_eye_dome_lighting()

add_mesh_kwargs = {'point_size': 2.0, 'opacity': 0.8, 'ambient': 0.5, 'show_scalar_bar': True}
plotter, _ = plot_point_clouds(noise_ptcld, 'gray', scalars = 'Z', **add_mesh_kwargs)
#plotter.screenshot(f'./Figures/{exp_name}.png')
plotter.show()
#%%
plotter = pv.Plotter(off_screen = False)
plotter.enable_eye_dome_lighting()

add_mesh_kwargs = {'point_size': 4.0, 'opacity': 0.8, 'ambient': 0.5, 'show_scalar_bar': True}
plotter, _ = plot_point_clouds(signal_ptcld, cmaps[idx], scalars = 'Z', **add_mesh_kwargs)
#plotter.screenshot(f'./Figures/{exp_name}.png')
plotter.show()

#%%
plotter = pv.Plotter(off_screen = False)
plotter.enable_eye_dome_lighting()

add_mesh_kwargs = {'point_size': 4.0, 'opacity': 0.8, 'ambient': 0.5, 'show_scalar_bar': True}
plotter, _ = plot_point_clouds(signal_ptcld, "glasbey_category10", scalars = 'Cluster ID', **add_mesh_kwargs)
#plotter.screenshot(f'./Figures/{exp_name}.png')
plotter.show()
#%%
#%%
hdbscan_params = {'min_samples': 1, 'cluster_selection_method': 'eom', 'cluster_selection_epsilon': 35}
desmin_denoised, _ = cluster_point_cloud(clipped_ptclouds[1], min_cluster_size = 20, **hdbscan_params)
actin_denoised, _ = cluster_point_cloud(clipped_ptclouds[0], min_cluster_size = 30, **hdbscan_params)
#%%
plotter = pv.Plotter(off_screen = False)
plotter.enable_eye_dome_lighting()

add_mesh_kwargs = {'opacity': 0.8, 'ambient': 0.5, 'show_scalar_bar': True, 'scalars': 'Z'}
plotter, _ = plot_point_clouds(actin_denoised, cmaps[0], plotter = plotter,  point_size = 2.0, **add_mesh_kwargs)
plotter, _ = plot_point_clouds(desmin_denoised, cmaps[1], plotter = plotter,  point_size = 4.0, **add_mesh_kwargs)
#plotter.screenshot(f'./Figures/{exp_name}.png')
plotter.camera_position = 'xy'
plotter.show()
# %%
foreground_inds = clusterer.labels_ != -1
foreground_pts = clipped_ptcloud[idx].coordinates[foreground_inds]
#%%

surf = desmin_denoised.pointcloud.reconstruct_surface()
surf
# %%
pl = pv.Plotter(shape=(1,2))
pl.add_mesh(desmin_denoised.pointcloud)
pl.add_title('Point Cloud of 3D Surface')
pl.subplot(0,1)
pl.add_mesh(surf, color=True, show_edges=True)
pl.add_title('Reconstructed Surface')
pl.show()
# %%
surf.plot(show_edges = True)
# %%
slices = desmin_denoised.pointcloud.slice_along_axis(n = 10)
slices.plot(line_width=5)
# %%
grid = pv.UniformGrid()
grid.origin = (6000, 2000, -320)
grid.spacing = (20, 20, 20)
grid.dimensions = (250, 250, 55)

p = pv.Plotter()
p.add_mesh(grid.outline(), color='k')
add_mesh_kwargs = {'opacity': 0.8, 'ambient': 0.5, 'show_scalar_bar': True}
p.add_mesh(desmin_denoised.pointcloud, scalars = 'Intensity', cmap = cmaps[1], **add_mesh_kwargs)
p.show()
# %%
interp = grid.interpolate(desmin_denoised.pointcloud, radius=15000, sharpness = 3, strategy='mask_points')
#%%
vol_opac = [0, 0, .2, 0.2, 0.5, 0.5]

p = pv.Plotter(shape=(1,2), window_size=[1024*3, 768*2])
p.enable_depth_peeling()
p.add_volume(interp, opacity=vol_opac)
p.add_mesh(desmin_denoised.pointcloud, scalars = 'Intensity', cmap = cmaps[1], point_size=4.0, **add_mesh_kwargs)
p.subplot(0,1)
p.add_mesh(interp.contour(5), opacity=0.5)
p.add_mesh(desmin_denoised.pointcloud, scalars = 'Intensity', cmap = cmaps[1], point_size=4.0, **add_mesh_kwargs)
p.link_views()
p.show()
# %%
