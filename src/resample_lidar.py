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
import scipy.ndimage.filters as filters
from scipy.ndimage import median_filter, labeled_comprehension
from collections import deque
from itertools import islice
from sklearn.neighbors import NearestNeighbors, KDTree
#import georasters as gr



#csv = '/home/mers/Bureau/teledm/donnees/lidar/calipso/caliop/2014_01_13.csv'
#df = pd.read_csv(csv)
#lat = df.Latitude.values
#lon = df.Longitude.values
#points = np.vstack((lon,lat)).T
#x = np.arange(-25,57.01,0.25)
#y = np.arange(-0.3,51.01,0.25)[::-1]
#sz = 11
#var_list = ["DEM_Surface_Elevation","Column_Optical_Depth_Aerosols_532","Feature_Optical_Depth_532","Feature_Optical_Depth_Uncertainty_532","Layer_Top_Altitude","Layer_Base_Altitude"]
#var = 'Layer_Base_Altitude'


def lissage_mediane(df_in,size,resamplelist,variable):
    dataframe = df_in.copy()
    dataframe[variable] = median_filter(dataframe[variable].values,size=size)
    nonmodif_idx = np.where(df_in[variable].values == dataframe[variable])[0]
    modif_idx = np.where(df_in[variable].values != dataframe[variable])[0]
    vlist = list(set(resamplelist) - set(variable))
    for v in vlist[:]:
        print v
        mat = dataframe[v].values[:]
        mat_out = np.zeros(mat.shape[0])
        mat_out[:] = np.nan
        for idx in modif_idx:
    #        print "\n idx : ",idx
    #        print "mat[idx] avant ",mat[idx]
    #        print "valeurs en entree ",df.Layer_Base_Altitude[idx-(sz//2):idx+(sz//2)+1].values
    #        print "valeurs pour la med : ",mat[idx-(sz//2):idx+(sz//2)+1]
            if idx-(size//2) < 0:
                diff_valeurs = np.abs(idx-(size//2) )
                mat_tmp = np.append(mat[:idx+(size//2)+1],mat[-diff_valeurs:])
                mat_out[idx] = np.median(mat_tmp)
            if idx+(size//2)+1 > mat.shape[0]:
                diff_valeurs = idx+(size//2)+1 - mat.shape[0]
                mat_tmp = np.append(mat[idx-(size//2):],mat[:diff_valeurs])
                mat_out[idx] = np.median(mat_tmp)
    #        print "median : ",np.median(mat[idx-(sz//2):idx+(sz//2)+1])
    #        print "median np : ",df.Layer_Base_Altitude_lisse_np.values[idx],'\n'
            mat_out[idx] = np.median(mat[idx-(size//2):idx+(size//2)+1])
            mat_out[nonmodif_idx] = mat[nonmodif_idx]
    #        print 'mat[idx] apres ',mat[idx]
        dataframe[v] = mat_out[:]
    return dataframe

def points2matrice(lon,lat,valeurs,x,y):
    matrice = np.zeros((y.shape[0],x.shape[0]))
    matrice[:] = np.nan
    for i in range(y.shape[0]):
        for j in range(x.shape[0]):
            idx = np.where((lat >= y[i]) & (lat < y[i]+0.25 ) & (lon >= x[j]) & (lon < x[j]+0.25))[0]
            if idx.size:
                matrice[i,j] = np.mean(valeurs[idx])
    return matrice

#dfs = map(lissage_mediane,(df,11,var_list,var))
#df11 = lissage_mediane(df,11,var_list,var)
#df9 = lissage_mediane(df,9,var_list,var)
#df15 = lissage_mediane(df,15,var_list,var)
#df21 = lissage_mediane(df,21,var_list,var)
#df31 = lissage_mediane(df,31,var_list,var)
#df31_31 = lissage_mediane(df31,31,var_list,'Layer_Top_Altitude')
#df9_9 = lissage_mediane(df9,9,var_list,'Layer_Top_Altitude')
#df15_15 = lissage_mediane(df15,15,var_list,'Layer_Top_Altitude')
#df21_21 = lissage_mediane(df21,21,var_list,'Layer_Top_Altitude')
#
#
#
#plt.figure(1)
#plt.subplot(2,3,1)
#plt.plot(df.Layer_Base_Altitude[:300],'k',label='Base')
#plt.plot(df9.Layer_Base_Altitude[:300],'--',label='Base (9)')
#plt.plot(df15.Layer_Base_Altitude[:300],':',label='Base (15)')
#plt.plot(df21.Layer_Base_Altitude[:300],'*',label='Base (21)')
#plt.plot(df31.Layer_Base_Altitude[:300],label='Base (31)')
#plt.legend()
#plt.subplot(2,3,4)
#plt.plot(df.Layer_Base_Altitude[30:100],'k',label='Base')
#plt.plot(df9.Layer_Base_Altitude[30:100],'--',label='Base (9)')
#plt.plot(df15.Layer_Base_Altitude[30:100],':',label='Base (15)')
#plt.plot(df21.Layer_Base_Altitude[30:100],'*',label='Base (21)')
#plt.plot(df31.Layer_Base_Altitude[30:100],label='Base (31)')
#plt.legend()
#plt.subplot(2,3,2)
#plt.plot(df.Layer_Top_Altitude[:300],'k',label='Top')
#plt.plot(df9.Layer_Top_Altitude[:300],'--',label='Top (9)')
#plt.plot(df15.Layer_Top_Altitude[:300],':',label='Top (15)')
#plt.plot(df21.Layer_Top_Altitude[:300],'*',label='Top (21)')
#plt.plot(df31.Layer_Top_Altitude[:300],label='Top (31)')
#plt.legend()
#plt.subplot(2,3,5)
#plt.plot(df.Layer_Top_Altitude[30:100],'k',label='Top')
#plt.plot(df9.Layer_Top_Altitude[30:100],'--',label='Top (9)')
#plt.plot(df15.Layer_Top_Altitude[30:100],':',label='Top (15)')
#plt.plot(df21.Layer_Top_Altitude[30:100],'*',label='Top (21)')
#plt.plot(df31.Layer_Top_Altitude[30:100],label='Top (31)')
#plt.legend()
#plt.subplot(2,3,3)
#plt.plot(df.Column_Optical_Depth_Aerosols_532[:300],'k',label='AOD')
#plt.plot(df9.Column_Optical_Depth_Aerosols_532[:300],'--',label='AOD (9)')
#plt.plot(df15.Column_Optical_Depth_Aerosols_532[:300],':',label='AOD (15)')
#plt.plot(df21.Column_Optical_Depth_Aerosols_532[:300],'*',label='AOD (21)')
#plt.plot(df31.Column_Optical_Depth_Aerosols_532[:300],label='AOD (31)')
#plt.legend()
#plt.subplot(2,3,6)
#plt.plot(df.Column_Optical_Depth_Aerosols_532[30:100],'k',label='AOD')
#plt.plot(df9.Column_Optical_Depth_Aerosols_532[30:100],'--',label='AOD (9)')
#plt.plot(df15.Column_Optical_Depth_Aerosols_532[30:100],':',label='AOD (15)')
#plt.plot(df21.Column_Optical_Depth_Aerosols_532[30:100],'*',label='AOD (21)')
#plt.plot(df31.Column_Optical_Depth_Aerosols_532[30:100],label='AOD (31)')
#plt.legend()
#plt.show()
#
#
#plt.figure(1)
#plt.subplot(2,3,1)
#plt.plot(df.Layer_Base_Altitude[:300],'k',label='Base')
#plt.plot(df9.Layer_Base_Altitude[:300],'--',label='Base (9)')
#plt.plot(df9_9.Layer_Base_Altitude[:300],'r',ls='dotted',lw=2.0,label='Base double filtre 9')
#plt.legend()
#plt.subplot(2,3,4)
#plt.plot(df.Layer_Base_Altitude[30:100],'k',label='Base')
#plt.plot(df9.Layer_Base_Altitude[30:100],'--',label='Base (9)')
#plt.plot(df9_9.Layer_Base_Altitude[30:100],'r',ls='dotted',lw=2.0,label='Base double filtre 9')
#plt.legend()
#plt.subplot(2,3,2)
#plt.plot(df.Layer_Top_Altitude[:300],'k',label='Top')
#plt.plot(df9.Layer_Top_Altitude[:300],'--',label='Top (9)')
#plt.plot(df9_9.Layer_Top_Altitude[:300],'r',ls='dotted',lw=2.0,label='Top double filtre 9')
#plt.legend()
#plt.subplot(2,3,5)
#plt.plot(df.Layer_Top_Altitude[30:100],'k',label='Top')
#plt.plot(df9.Layer_Top_Altitude[30:100],'--',label='Top (9)')
#plt.plot(df9_9.Layer_Top_Altitude[30:100],ls='dotted',lw=2.0,label='Top double filtre 9')
#plt.legend()
#plt.subplot(2,3,3)
#plt.plot(df.Column_Optical_Depth_Aerosols_532[:300],'k',label='AOD')
#plt.plot(df9.Column_Optical_Depth_Aerosols_532[:300],'--',label='AOD (9)')
#plt.plot(df9_9.Column_Optical_Depth_Aerosols_532[:300],ls='dotted',lw=2.0,label='AOD double filtre')
#plt.legend()
#plt.subplot(2,3,6)
#plt.plot(df.Column_Optical_Depth_Aerosols_532[30:100],'k',label='AOD')
#plt.plot(df9.Column_Optical_Depth_Aerosols_532[30:100],'--',label='AOD (9)')
#plt.plot(df9_9.Column_Optical_Depth_Aerosols_532[30:100], ls='dotted',lw=2.0,label='AOD double filtre 9')
#plt.legend()
#plt.show()
#
#fig,axes = plt.subplots(3,6)
#fig.patch.set_facecolor('white')
#axes = axes.ravel()
#vlisse = ['DEM_Surface_Elevation_lisse',
# 'Column_Optical_Depth_Aerosols_532_lisse',
# 'Feature_Optical_Depth_532_lisse',
# 'Feature_Optical_Depth_Uncertainty_532_lisse',
# 'Layer_Top_Altitude_lisse','Layer_Base_Altitude_lisse']*3
#vinit = resamplelist*3
#
#for i in range(6):
#    axes[i].plot(df[vinit[i]].values,label=vinit[i])
#    axes[i].plot(df[vlisse[i]].values, 'k--', label=vlisse[i])
#    axes[i].legend()
#for i in range(6,12,1):
#    axes[i].plot(df[vinit[i]].values[:300],label=vinit[i])
#    axes[i].plot(df[vlisse[i]].values[:300], 'k--', label=vlisse[i])
#    axes[i].legend()
#for i in range(12,18,1):
#    axes[i].plot(df[vinit[i]].values[30:80],label=vinit[i])
#    axes[i].plot(df[vlisse[i]].values[30:80], 'k--', label=vlisse[i])
#    axes[i].legend()
#plt.show()






#mat = points2matrice(lon,lat,df.Layer_Base_Altitude.values,x,y)
#
#plt.imshow(mat, extent=[-25.,57.01,-0.3,51.01], interpolation='none')
#plt.colorbar()
#plt.plot(points[:,0],points[:,1], 'k.',ms=1)
#plt.show()
#
#plt.plot(df.Layer_Base_Altitude.values[:300],label='base')
#plt.plot(mat_out[:300],'k.',label='base_lisse')
#plt.plot(df.Layer_Top_Altitude.values[:300], label='top')
#plt.plot(df.Column_Optical_Depth_Aerosols_532.values[:300], label='AOD_532')
#plt.plot(df.DEM_Surface_Elevation.values[:300], label='DEM')
#plt.plot(df.Feature_Optical_Depth_532.values[:300], label='Feature_OD_532')
#plt.plot(df.Feature_Optical_Depth_Uncertainty_532.values[:300], label='Uncertainty_532')
#plt.plot(df.Layer_Top_Altitude_lisse.values[:300],'k--', label='top_lisse')
#plt.plot(df.Column_Optical_Depth_Aerosols_532_lisse.values[:300], label='AOD_532_lisse')
#plt.plot(df.DEM_Surface_Elevation_lisse.values[:300], label='DEM_lisse')
#plt.plot(df.Feature_Optical_Depth_532_lisse.values[:300], label='Feature_OD_532_lisse')
#plt.plot(df.Feature_Optical_Depth_Uncertainty_532_lisse.values[:300], label='Uncertainty_532_lisse')
#plt.legend(),plt.show()
