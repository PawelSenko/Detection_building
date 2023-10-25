# %%
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.transform import Affine
from rasterio.plot import show
from rasterio.plot import show_hist
import numpy as np
from matplotlib import pyplot as plt
import earthpy
from rasterio.mask import mask
import pandas as pd  

# %%
img = rasterio.open('mapy/mapa5-10cm-d2-2.tif') #zmienić nazwę
mapa = img.read(1)





# %%
import geopandas as gpd
df = gpd.read_file('budynki_szczecin.geojson')

df.head()
#gdf = df.drop('bbox_20m', axis=1)
#gdf = gdf.drop('centroid', axis=1)
#df['bbox_20m']

# %%
from shapely.geometry import MultiPolygon, Polygon
from shapely import wkt

col_bbox = df['bbox_20m']
col_load= wkt.loads(col_bbox)
col_multi = col_load.apply(lambda x: MultiPolygon([x]))
print(col_multi)

# %%
df['bbox_20m'] = col_multi
df.head()


# %%
gdf = df.loc[df['x_kod']=='BUBD07']


bounds=img.bounds
print(bounds)
gdf_bbox = gdf.cx[bounds.left:bounds.right,bounds.bottom:bounds.top]
#gdf2 = gdf.cx[203171.9:205390.8,626019.8:628464.7]

print(len(gdf_bbox))
gdf_bbox.head()

# %%
spis = []

for i in range(21):
    if i < 9 :
        gdf = df.loc[df['x_kod']==f'BUBD0{i+1}']
        gdf2 = gdf.cx[bounds.left:bounds.right,bounds.bottom:bounds.top]
        number = len(gdf2)
        spis.append(number)
    else :
        gdf = df.loc[df['x_kod']==f'BUBD{i+1}']
        gdf2 = gdf.cx[bounds.left:bounds.right,bounds.bottom:bounds.top]
        number = len(gdf2)
        spis.append(number)
print(spis)

# %%
gdf3 = gdf_bbox['bbox_20m']
#print(gdf3)
n =len(gdf3)
print(n)
gdf5 = gdf3.iloc[25]
print('5:',gdf5)

bbox = gdf5.bounds
print('bbox',bbox)
window1 = img.window(*bbox)
print('window',window1)
mapa1 = img.read(1,window=window1)
#plt.imshow(mapa1)
plt.imshow(mapa1, extent=bbox)
#gdf5.boundary.plot()

#gdf5.boundary.plot(ax=plt.gca(), color='skyblue')
#gdf_07.boundary.plot(ax=plt.gca(), color='skyblue')


# %%
print(gdf5)
bbox3 = gdf5.bounds
print(bbox3)
windo3 = img.window(*bbox3)
print(windo3)
# Read the windowed region of the raster data
raster_data = img.read(window=windo3)

# Specify the transform for the windowed region
window_transform = img.window_transform(windo3)

# Plot the windowed region
fig, ax = plt.subplots()
show(raster_data, ax=ax, transform=window_transform)

# Display the windowed region
plt.axis('off')
plt.savefig("test.png",bbox_inches='tight',pad_inches = 0)
plt.show()


# %%
import multiprocessing
from multiprocessing import Process
from functools import wraps,cache
from numba import jit, cuda, njit
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
from functools import wraps

def memoize(func):
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)

        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

print("Number of cpu : ", multiprocessing.cpu_count())

# %%
Buildings = ['BUBD01','BUBD03','BUBD04','BUBD05','BUBD07','BUBD08','BUBD10','BUBD11','BUBD12','BUBD15','BUBD16','BUBD18']
n_build = len(Buildings)
gdf_all = gdf_bbox['bbox_20m']

bounds=img.bounds
print(bounds)

@cache
def loop_build(): 
    for i in Buildings:
        print(i)
        gdf = df.loc[df['x_kod']==i]
        gdf_bbox = gdf.cx[bounds.left:bounds.right,bounds.bottom:bounds.top]
        gdf_all = gdf_bbox['bbox_20m']
        n_all = len(gdf_all)
        print(n_all)
        for j in range(n_all):
            #print(j)
            gdf_loop = gdf_all.iloc[j]
            #print(gdf_loop)
            bbox_loop = gdf_loop.bounds
            window_loop = img.window(*bbox_loop)
            raster_data_loop = img.read(window=window_loop)
            window_transform_loop = img.window_transform(window_loop)
            fig, ax = plt.subplots()
            show(raster_data_loop, ax=ax, transform=window_transform_loop)

            # Display the windowed region
            plt.axis('off')
            plt.savefig(f"data/{i}/{i}_d2-2_{j+1}.png",bbox_inches='tight',pad_inches = 0)
            plt.close()

    
if __name__=="__main__":
    loop_build()
    print("finish")
  

# %%
for i in range (n):
    print(i)
    gdf_loop = gdf3.iloc[i]
    #print(gdf_loop)
    bbox_loop = gdf_loop.bounds
    window_loop = img.window(*bbox_loop)
    raster_data_loop = img.read(window=window_loop)
    window_transform_loop = img.window_transform(window_loop)
    fig, ax = plt.subplots()
    show(raster_data_loop, ax=ax, transform=window_transform_loop)

    # Display the windowed region
    plt.axis('off')
    plt.savefig(f"data/BUBD07/BUBD07_a34-{i+1}.png",bbox_inches='tight',pad_inches = 0)
    #plt.show()

# %%
mapa_transform = img.window_transform(window1)



with rasterio.open('07mapa.tif', #filename
                   'w', # file mode, with 'w' standing for "write"
                   driver='GTiff', # format to write the data
                   height=mapa1.shape[0], # height of the image, often the height of the array
                   width=mapa1.shape[1], # width of the image, often the width of the array
                   count=1, # the number of bands to write
                   dtype=rasterio.ubyte, # the dtype of the data, usually `ubyte` if data is stored in integers
                   crs=img.crs, # the coordinate reference system of the data
                   transform=mapa_transform # the affine transformation for the image
                  ) as outfile:
    outfile.write(mapa1,indexes=2) # write the `austin_nightlights` as the first band


# %%
import fiona
import rasterio
import rasterio.mask
from rasterio.windows import Window
from pyproj import Transformer


# To mask the data

with fiona.open("budynki_szczecin2.geojson", "r") as shapefile:
  for feature in shapefile:
    shapes = [feature['geometry']]
    with rasterio.open("mapa10cm.tif") as src:
      out_image, transformed = rasterio.mask.mask(src, shapes, crop=True)
      out_meta = src.meta 

# %%
show(out_image)

# %%
import contextily

basemap, basemap_extent = contextily.bounds2img(*gdf_07.to_crs(epsg=3857).total_bounds, 
                                                zoom=10)


plt.figure(figsize=(10,10))
plt.imshow(img)
gdf_07.to_crs(epsg=3857).plot('hot', 
                                      ax = plt.gca(), alpha=0)
plt.axis(gdf_07.to_crs(epsg=3857).total_bounds[[0,2,1,3]])


