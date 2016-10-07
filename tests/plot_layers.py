# -*- coding: utf-8 -*-

import pandas as pd
import os
import matplotlib.pyplot as plt


os.chdir('/home/mers/code/python/lidar_traitement/out')

files = '/home/mers/code/python/lidar_traitement/out/2014_01_27_16d_test.csv'
df = pd.read_csv(files, header=0)
df.Layer_Base_Extended[df.Layer_Base_Extended > 0] = 10

plt.plot(df.index, df.Layer_Base_Altitude_lissage2, label='Base')
plt.plot(df.index, df.Base_corr_lissage2, label='Base_corr')
plt.plot(df.index, df.Layer_Top_Altitude_lissage2, label='Top')
plt.plot(df.index, df.Top_corr_lissage2, label='Top_corr')
plt.plot(df.index, df.DEM_Surface_Elevation, label='DEM')
plt.plot(df.index, df.Layer_Base_Extended, "*", label='Layer Extended')

plt.show(), plt.legend()
