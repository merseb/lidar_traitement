# -*- coding: utf-8 -*-

import numpy as np
import os, sys
from pyhdf.SD import SD, SDC
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

ddir_fig =os.path.expanduser('~') +'/Bureau/teledm/fusion_donnees/resultats/figures'
path = os.path.expanduser('~')+"/code/python/lidar_traitement"
sys.path.append(path+"/src")
from rolling_window import *
from LidarUtil import *
os.chdir(path+'/2014/zone_etude')
matplotlib.use('Agg')
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

def readVariable(hdf, variable):
    layerInit = hdf.select(variable)
    layerMin = float(layerInit.attributes()['valid_range'].split('...')[0])
    layerMax = float(layerInit.attributes()['valid_range'].split('...')[1])
    matrice = layerInit[:]
    print layerMin
    layerInit.endaccess()
    matrice[matrice < layerMin] = np.nan
    matrice[matrice > layerMax] = np.nan
    return matrice

f = sys.argv[1]
#f = 'CAL_LID_L2_05kmALay-Prov-V3-30.2014-03-24T13-45-37ZD.hdf'
#f = 'CAL_LID_L2_05kmALay-Prov-V3-30.2014-02-08T00-57-04ZN.hdf'
date = f[31:41]
hdf = SD(f, SDC.READ)
lat = hdf.select('Latitude')[:, 1]
lon = hdf.select('Longitude')[:, 1]
baseInit = readVariable(hdf, 'Layer_Base_Altitude')
topInit = readVariable(hdf, 'Layer_Top_Altitude')
DEM = readVariable(hdf, 'DEM_Surface_Elevation')[:,2]
nbLayers = hdf.select("Number_Layers_Found")[:]
featInit = hdf.select('Feature_Classification_Flags')[:]
feature = np.array([decodeFeatureMask1(int16) for int16 in featInit.flatten().tolist()]).reshape(featInit.shape)
baseExtInit = hdf.select('Layer_Base_Extended')[:]
featureExt = np.array([decodeFeatureMask1(int(e)) for e in baseExtInit.flatten().tolist()]).reshape(baseExtInit.shape)
hdf.end()


base = baseInit - np.tile(DEM, baseInit.shape[1]).reshape((baseInit.shape[1],-1)).T
top = topInit - np.tile(DEM, baseInit.shape[1]).reshape((baseInit.shape[1],-1)).T
#fig, ax = plt.subplots(1)
#fig.title('2014 Base 1ere couche detectee')
#ax.plot(lat, baseInit[:,0], 'b', label='Base Init')
#ax.plot(lat, base[:,0], 'g:', label='Base Corr')
#ax.plot(lat, DEM, 'k', label='DEM')
#plt.grid()
#ax.legend(),plt.show()





### Dust
maskDust = feature == types['dust']
baseDust = np.ma.array(data=base, mask=~maskDust, fill_value=np.nan)
topDust = np.ma.array(data=top, mask=~maskDust, fill_value=np.nan)

maskExtDust = featureExt == types['dust']
baseExtDust = np.ma.array(data=base, mask=~maskExtDust, fill_value=np.nan)
topExtDust = np.ma.array(data=top, mask=~maskExtDust, fill_value=np.nan)

maskD = ~maskDust & ~maskExtDust
baseDust1 = np.ma.array(data=base, mask=maskD, fill_value=np.nan)
topDust1 = np.ma.array(data=top, mask=maskD, fill_value=np.nan)

listMat = np.vsplit(np.ma.filled(baseDust1, -9999), baseDust1.shape[0])  # decoupage de la matrice 1D en une liste de n sous-matrices de dim (1,8)
output = map(indiceCouche1, listMat)
indices = [m[1] for m in output]
baseDust0 = np.array([m[0] for m in output])
baseDust0[baseDust0==-9999] = np.nan
topDust0 = np.array([topDust1[i, indices[i]] for i in range(topDust1.shape[0])])



### Polluted Dust
maskPdust = feature == types['polluted_dust']
basePdust = np.ma.array(data=base, mask=~maskPdust, fill_value=np.nan)
topPdust = np.ma.array(data=top, mask=~maskPdust, fill_value=np.nan)

maskExtPdust = featureExt == types['polluted_dust']
baseExtPdust = np.ma.array(data=base, mask=~maskExtPdust, fill_value=np.nan)
topExtPdust = np.ma.array(data=top, mask=~maskExtPdust, fill_value=np.nan)

maskDp = ~maskPdust & ~maskExtPdust
basePdust1 = np.ma.array(data=base, mask=maskDp, fill_value=np.nan)
topPdust1 = np.ma.array(data=top, mask=maskDp, fill_value=np.nan)

listMat1 = np.vsplit(np.ma.filled(basePdust1, -9999), basePdust.shape[0])  # decoupage de la matrice 1D en une liste de n sous-matrices de dim (1,8)
output1 = map(indiceCouche1, listMat1)
indices1 = [m[1] for m in output]
basePdust0 = np.array([m[0] for m in output1])
basePdust0[basePdust0==-9999] = np.nan
topPdust0 = np.array([topPdust1[i, indices1[i]] for i in range(topPdust1.shape[0])])


#######################################################################################
fig = plt.figure(1, figsize=(15,10))
fig.suptitle(f + '\n')
gs = gridspec.GridSpec(7,1, hspace=0.1)
ax1 = plt.subplot(gs[:3, :])
ax2 = plt.subplot(gs[4:, :], sharex=ax1)
ax3 = plt.subplot(gs[3], sharex=ax1)
for i in range(8):
    ax1.fill_between(lat, baseDust[:,i], topDust[:,i], color="green", alpha=0.5)
    ax1.fill_between(lat, baseExtDust[:,i], topExtDust[:,i], color="blue", alpha=0.5)
    ax1.xaxis.grid(True)
    ax1.tick_params(axis='x', which='both', top='on', bottom='off', labeltop='on', labelbottom='off')
    ax1.set_ylabel('Altitude(km)')
    ax1.text(0.5, 0.95, 'Dust', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='green', bbox=dict(facecolor='none', edgecolor='green', pad=10.0))
    ax1.text(0.5, 0.85, 'Dust Layer Base Extended', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='blue', bbox=dict(facecolor='none', edgecolor='blue', pad=10.0))
    ax2.set_xlabel('Latitude')
    ax2.text(0.5, -0.45, 'Polluted Dust', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='green', bbox=dict(facecolor='none', edgecolor='green', pad=10.0))
    ax2.text(0.5, -0.55, 'Polluted Dust Layer Base Extended', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='blue', bbox=dict(facecolor='none', edgecolor='blue', pad=10.0))
    ax2.fill_between(lat, basePdust[:,i], topPdust[:,i], color="green", alpha=0.5)
    ax2.fill_between(lat, baseExtPdust[:,i], topExtPdust[:,i], color="blue", alpha=0.5)
    ax2.xaxis.grid(True)
    ax3.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
    ax2.set_ylabel('Altitude(km)')
    ax3.plot(lat, np.ma.array(nbLayers, mask=nbLayers==0), color='black', marker='.', ls='none', markeredgecolor='none', markerfacecolor='black')
    ax3.xaxis.grid(True)
    ax3.set_ylabel('nb Layers')
    ax3.set_yticklabels([1,'',2,'',3,'',4])
ax1.plot(lat, baseDust0, 'r:')
ax1.plot(lat, topDust0, 'r:')
ax2.plot(lat, basePdust0, 'r:')
ax2.plot(lat, topPdust0, 'r:')
fig.show()
plt.show()


#fig = plt.figure(1, figsize=(17,12))
#fig.suptitle(f)
#gs = gridspec.GridSpec(8,1, hspace=0.05)
#ax1 = plt.subplot(gs[:4, :])
#ax2 = plt.subplot(gs[4:], sharex=ax1)
#for i in range(8):
#    ax1.fill_between(lat, baseDust[:,i], topDust[:,i], color="blue", alpha=0.5)
#    #ax1.fill_between(lat, baseExtDust[:,i], topExtDust[:,i], color="green", alpha=0.5)
#    #ax1.fill_between(lat, baseDust1[:,i], topDust1[:,i], color="blue", alpha=0.5)
#    ax1.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
#    ax1.xaxis.grid(True)
#    ax1.set_ylabel('Altitude(km)')
#    #ax2.plot(lat, nbLayers, color='red', marker='.', ls='none', markeredgecolor='red', markerfacecolor='none')
#    #ax2.fill_between(lat, baseDust1[:,i], topDust1[:,i], color="blue", alpha=0.5)
#    ax2.fill_between(lat, baseExtDust[:,i], topExtDust[:,i], color="green", alpha=0.5)
#    ax2.xaxis.grid(True)
#    ax2.grid(True,'minor',color='k', alpha=0.2, ls='-', lw=0.2)
#    ax2.set_xlabel('Latitude')
#    ax2.set_ylabel('nb Layers')
#fig.show()
#
#
#fig = plt.figure(1, figsize=(17,12))
#fig.suptitle(f)
#gs = gridspec.GridSpec(7,1, hspace=0.05)
#ax1 = plt.subplot(gs[:6, :])
#ax2 = plt.subplot(gs[6], sharex=ax1)
#for i in range(8):
#    ax1.fill_between(lat, baseDust[:,i], topDust[:,i], color="k", alpha=0.5)
#    #ax1.fill_between(lat, baseExtDust[:,i], topExtDust[:,i], color="green", alpha=0.5)
#    #ax1.fill_between(lat, baseDust1[:,i], topDust1[:,i], color="blue", alpha=0.5)
#    ax1.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
#    ax1.xaxis.grid(True)
#    ax1.set_ylabel('Altitude(km)')
#    ax1.text(45, 38, 'dust', style='italic',bbox={'facecolor':'green', 'alpha':0.5, 'pad':10})
#    ax1.text(45, 35, 'polluted dust', style='italic',bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
#
#    ax2.plot(lat, nbLayers, color='red', marker='.', ls='none', markeredgecolor='red', markerfacecolor='none')
#    ax2.xaxis.grid(True)
#    ax2.set_xlabel('Latitude')
#    ax2.set_ylabel('nb Layers')
#    ax2.set_yticklabels([1,'',2,'',3,'',4])
#fig.show()