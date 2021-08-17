#%%
import numpy as np
from aicsimageio import AICSImage

img = AICSImage('./Data/Fed_X63_Z3_SIM.czi')
# %%
img.data
# %%
scale = tuple(img.physical_pixel_sizes)
# %%
data_dict = {'data': img.data, 'scale': scale, 'channels': ('actin', 'desmin')}
# %%
np.savez('./Data/Fed_X63_Z3_SIM.npz', **data_dict)
# %%
