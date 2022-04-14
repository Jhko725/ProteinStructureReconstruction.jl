#%%
import numpy as np
from aicsimageio import AICSImage

czi_file = AICSImage('./Data/SIM2_desmin_alpha actinin 25Jan22/Image 11_SIM2.czi')
#%%
czi_file.xarray_data
#%%

#czi_file = AICSImage('./Data/Fed_X63_Z3_SIM.czi')
dim_order = 'CZYX'
img_data = czi_file.get_image_data(dim_order, T = 0)
channel_names = ['desmin', 'alpha-actinin'] 
# %%
import h5py
with h5py.File('./Data/Image11_SIM2.h5', 'w') as f:
    dataset = f.create_dataset("Data", data = img_data)
    dataset.attrs['dim_order'] = dim_order
    dataset.attrs['channels'] = channel_names
    dataset.attrs['pixel_sizes'] = np.array(czi_file.physical_pixel_sizes, dtype = np.float32)

# %%
import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(np.mean(img_data[1], axis = 0))
# %%
