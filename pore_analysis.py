#%%
import numpy as np

data = np.load('./Data/interp.npz')
desmin, actin = data['desmin'], data['actin']
# %%
desmin.shape
# %%
import porespy
out = porespy.filters.local_thickness(desmin>0.1)
# %%
out
# %%
import plotly.graph_objects as go
X, Y, Z = np.meshgrid(*[np.arange(N) for N in out.shape])
#%%
fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=out.flatten(),
    isomin=0.1,
    isomax=10,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=17, # needs to be a large number for good volume rendering
    ))
fig.show()
# %%
import matplotlib.pyplot as plt
psd = porespy.metrics.pore_size_distribution(out)
%matplotlib inline
plt.bar(psd.bin_centers, psd.pdf, width=psd.bin_widths, edgecolor='k')
# %%
np.min(desmin)
# %%
np.unique(desmin)
# %%
psd.__dict__
# %%
