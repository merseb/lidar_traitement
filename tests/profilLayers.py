# -*- coding: utf-8 -*-

import numpy as np
import os, sys
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

ddir_fig =os.path.expanduser('~') +'/Bureau/teledm/fusion_donnees/resultats/figures'
path = os.path.expanduser('~')+"/code/python/lidar_traitement"
sys.path.append(path+"/src")
from rolling_window import *
from LidarUtil import *
os.chdir(path+'/zone_etude')

# liste des variables/parametres extraits de chaque fichier lidar
varlist = ["IGBP_Surface_Type",
           "Day_Night_Flag",
           "DEM_Surface_Elevation",
           "Column_Optical_Depth_Aerosols_532",
           "Feature_Optical_Depth_532",
           "Feature_Optical_Depth_Uncertainty_532",
           "ExtinctionQC_532",
           "CAD_Score",
           "Feature_Classification_Flags",
           "Number_Layers_Found",
           "Layer_Base_Extended",
           "Relative_Humidity",
           "Layer_Top_Altitude",
           "Layer_Base_Altitude"]

# Variables/parametres sur lesquelles lissees
varLissage = ['Base_corr', 'Top_corr', 'Column_Optical_Depth_Aerosols_532',  'Concentration_Aerosols']   ##, 'Feature_Optical_Depth_532', 'Feature_Optical_Depth_Uncertainty_532', 'ExtinctionQC_532', 'Relative_Humidity']
subtypes = ['dust', 'polluted_dust']
types = {"clean_marine":1, "dust":2, "polluted_continental":3, "clean_continental":4, "polluted_dust":5, "smoke":6}



f = 'CAL_LID_L2_05kmALay-Prov-V3-30.2014-03-24T13-45-37ZD.hdf'
#f = 'CAL_LID_L2_05kmALay-Prov-V3-30.2014-02-08T00-57-04ZN.hdf'
date = f[31:41]
hdf = SD(f, SDC.READ)
#var = sorted(hdf.datasets().keys())
#for i in range(len(var)):
#    vr = hdf.select(var[i])
#    print '[%i] %s  dims (%d, %d), units: %s, valid_range: %s, format: %s' % (i, var[i], vr[:].shape[0], vr[:].shape[1], vr.attributes()['units'], vr.attributes()['valid_range'],vr.attributes()['format'])
#    vr.endaccess()
        
lat = hdf.select('Latitude')[:, 1]
lon = hdf.select('Longitude')[:, 1]
baseInit = hdf.select('Layer_Base_Altitude')[:]
nbLayers = hdf.select("Number_Layers_Found")[:].astype(float)
nbLayers[nbLayers==0] = np.nan
featIn = hdf.select('Feature_Classification_Flags')[:]
feature = np.array([decodeFeatureMask1(int16) for int16 in featIn.flatten().tolist()]).reshape(featIn.shape)
layerExt = hdf.select('Layer_Base_Extended')[:].astype(float)
layerExt[(layerExt==0) | (layerExt==1)] = np.nan
mdust = feature.copy().astype(float)
mdust[mdust != types['dust']] = np.nan
DEM = hdf.select('DEM_Surface_Elevation')[:, 2]# * mdust
bdust = baseInit * mdust
baseDust = (bdust.flatten() - np.tile(DEM,8)).reshape((-1,8))
baseDust[baseDust<0] = np.nan
listMat = np.vsplit(baseInit, baseInit.shape[0])  # decoupage de la matrice 1D en une liste de n sous-matrices de dim (1,8)
output = map(indiceCouche1, listMat)
base = np.array([m[0] for m in output])
baseCorr = base - DEM
baseCorr[baseCorr<0] = np.nan
indices = [m[1] for m in output]
tdust = hdf.select('Layer_Top_Altitude')[:] * mdust
topDust = (tdust.flatten() - np.tile(DEM,8)).reshape((-1,8))
topDust[topDust<0] = np.nan
top = np.array([topDust[i, indices[i]] for i in range(topDust.shape[0])])
topCorr = top - DEM
topCorr[topCorr<0] = np.nan

mpdust = feature.copy().astype(float)
mpdust[mpdust != types['polluted_dust']] = np.nan
bpdust = baseInit * mpdust
basePdust = (bpdust.flatten() - np.tile(DEM,8)).reshape((-1,8))
basePdust[basePdust<0] = np.nan
tpdust = hdf.select('Layer_Top_Altitude')[:] * mpdust
topPdust = (bpdust.flatten() - np.tile(DEM,8)).reshape((-1,8))
topPdust[topPdust<0] = np.nan

fig = plt.figure(1, figsize=(17,12))
fig.suptitle(f)
gs = gridspec.GridSpec(7,1, hspace=0.1)
ax1 = plt.subplot(gs[:3, :])
ax2 = plt.subplot(gs[3:6, :], sharex=ax1)
ax3 = plt.subplot(gs[6], sharex=ax1)
for i in range(8):
    ax1.fill_between(lat, baseDust[:,i], topDust[:,i], color="green", alpha=0.5)
    ax1.plot(lat, baseCorr, 'g-')
    ax1.plot(lat, topCorr, 'r-')
    ax1.xaxis.grid(True)
    ax1.set_ylabel('Altitude(km)')
    ax1.text(45, 38, 'dust', style='italic',bbox={'facecolor':'green', 'alpha':0.5, 'pad':10})
    ax1.text(45, 35, 'polluted dust', style='italic',bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
    ax2.fill_between(lat, basePdust[:,i], topPdust[:,i], color="blue", alpha=0.5)
    ax2.xaxis.grid(True)
    ax2.set_ylabel('Altitude(km)')
    ax3.plot(lat, nbLayers, color='red', marker='.', ls='none', markeredgecolor='red', markerfacecolor='none')
    ax3.xaxis.grid(True)
    ax3.set_xlabel('Latitude')
    ax3.set_ylabel('nb Layers')
fig.show()