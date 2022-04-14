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
exp_name = 'desmin_alphaactinin_600nm'
#exp_name = 'actin_desmin_600nm'
cmaps = ['kbc', 'fire']
exp_filepath = fetch_STORM_experiment_filepath(exp_name, STORM_data_dir)
storm_df = read_Vutara352_localization_data(exp_filepath)
storm_df = drop_invalid_localizations(storm_df)

storm_df.to_csv("./Data/STORM/desmin_alphaactinin_600nm.csv")
# 
# probe_ids, storm_dfs = get_per_probe_localizations(storm_df)
#%%
storm_df.head()
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize = (7, 5))
ax.hist(storm_dfs[1].loc[:, 'valid'], bins = 100)

# %%
