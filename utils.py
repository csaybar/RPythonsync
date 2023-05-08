import pathlib
import numpy as np
import rasterio as rio

def create_memmap(file, shape):
    binfile = np.memmap(file, dtype='float32', mode='w+', shape=shape)
    rfiles = list(pathlib.Path("data/").glob("*.tif"))
    for index, file in enumerate(rfiles):
        with rio.open(file) as src:
            data = src.read()/10000
        # padding to 512x512
        data = np.pad(data, ((0,0),(0,1),(0,1)), 'constant', constant_values=0)
        binfile[index] = data
    binfile.flush()
    return True
