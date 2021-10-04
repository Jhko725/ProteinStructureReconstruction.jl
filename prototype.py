#%%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from skimage.io import imread
from skimage.color import rgb2gray

images = list(map(lambda x: np.float32(rgb2gray(imread(x))), ['./Data/SR2_desmin_image.jpg', './Data/SR3_actin_image.jpg']))
#%%
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
ax.hist(images[0].flatten(), bins = 40)
ax.set_yscale('log', base = 10)
# %%
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
ax.imshow(np.stack([images[0][-2987:, -3004:], images[1], np.zeros_like(images[1])], axis = -1))

# %%
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
points = images[0]>0
ax.imshow(points[2900:3000, 2900:3000])
# %%
def convert_to_pointcloud(image):
    xx, yy = np.meshgrid(*[np.arange(n) for n in image.shape], indexing = 'ij')
    inds = image > 0
    return np.stack((xx[inds], yy[inds]), axis = -1)
# %%

# %%
from functools import reduce
from astropy.stats import RipleysKEstimator
img = images[0][2900:3000, 2900:3000]
coords = convert_to_pointcloud(img)
Kest = RipleysKEstimator(area = reduce(lambda x, y: x*y, img.shape))
r = np.linspace(0, 100, 100)
result = Kest(data=coords, radii=r, mode='none')
# %%
result
# %%
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
ax.plot(r, result)
# %%

# %%
#import plotly.io as pio
#pio.renderers.default = "vscode"
import plotly.graph_objects as go
fig = go.Figure(data = go.Scattergl(x = images[0][-2987:, -3004:].flatten(), y = images[1].flatten()))
fig.show()

# %%
np.max(coords)
# %%
images[1].shape
# %%
images[0][-2987:, -3004:].shape
# %%


# %%
