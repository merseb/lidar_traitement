# -*- coding: utf-8 -*-


from netCDF4 import Dataset, num2date, date2num, date2index
import numpy as np
import pandas as pd
from pyresample import geometry, kd_tree
from glob import glob
import os
from matplotlib.mlab import griddata as grdata
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpldatacursor import datacursor
import scipy 
from scipy import ndimage
from collections import deque
from itertools import islice
from sklearn.neighbors import NearestNeighbors, KDTree
import georasters as gr


csv = '/work/crct/se5780me/lidar/2014/zone_etude/2014_01_13.csv'
df = pd.read_csv(csv)
lat = df.Latitude.values
lon = df.Longitude.values
lba = df.Layer_Base_Altitude.values

fnc = '/work/crct/se5780me/teledm_toolbox/teledm_update/telechargement/seviri_aerus/seviri_r005_14d.nc'
nc = Dataset(fnc,'r')
aod = nc.variables['AOD_VIS06'][:]


x = np.arange(-25,57.01,0.25)
y = np.arange(-0.3,51.01,0.25)[::-1]
matrice = np.zeros((y.shape[0],x.shape[0]))
matrice[:] = np.nan
for i in range(5,y.shape[0],1):
	for j in range(5,x.shape[0],1):
		idx = np.where((lat >= y[i-5]) & (lat < y[i+6] ) & (lon >= x[j-5]) & (lon < x[j+6]))[0]
		print idx


