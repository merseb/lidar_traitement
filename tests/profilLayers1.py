# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os, sys
from pyhdf.SD import SD, SDC
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

ddir_fig =os.path.expanduser('~') +'/Bureau/teledm/fusion_donnees/resultats/figures'
path = os.path.expanduser('~')+"/code/python/lidar_traitement"
sys.path.append(path+"/src")
#from rolling_window import *
from LidarUtil import *
#os.chdir(path+'/2014/zone_etude')

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
    if layerInit.attributes()['format'] == 'Float_32':
        layerMin = float(layerInit.attributes()['valid_range'].split('...')[0])
        try:
            layerMax = float(layerInit.attributes()['valid_range'].split('...')[1])
        except ValueError:
            layerMax = np.inf
        matrice = layerInit[:]
        layerInit.endaccess()
        matrice[matrice < layerMin] = np.nan
        matrice[matrice > layerMax] = np.nan
    return matrice


def medFilt(x, k):
    k2 = k // 2
    y = np.zeros ((len (x), k*2), dtype=x.dtype)
    y[:,k2] = x
    indices = range(x.shape[0])
    y[:,k2+k] = indices
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)-k] = x[j:]
        y[-j:,-(i+1)-k] = x[-1]
        y[j:,i+k] = indices[:-j]
        y[:j,i+k] = indices[0]
        y[:-j,-(i+1)] = indices[j:]
        y[-j:,-(i+1)] = indices[-1]
    args = np.argsort(y[:,:k], axis=1)[:,k2]
    return np.median (y[:,:k], axis=1), y[:,k:][indices,args].astype(int)



#f = sys.argv[1]
f = path + '/2014/zone_etude/CAL_LID_L2_05kmALay-Prov-V3-30.2014-03-24T13-45-37ZD.hdf'
#f = 'CAL_LID_L2_05kmALay-Prov-V3-30.2014-02-08T00-57-04ZN.hdf'

if not f:
    print "nom de fichier manquant"
    sys.exit()

date = f[31:41]
try:
    hdf = SD(f, SDC.READ)
except Exception as ex:
    print ex
    sys.exit()
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
CAD = hdf.select("CAD_Score")
cadMin = int(CAD.attributes()['valid_range'].split('...')[0])
cadMax = int(CAD.attributes()['valid_range'].split('...')[1])
cad = CAD[:]
CAD.endaccess()
qc = hdf.select("ExtinctionQC_532")[:]
aodUnc = readVariable(hdf, "Feature_Optical_Depth_Uncertainty_532")
aod = readVariable(hdf, "Feature_Optical_Depth_532")
caod = readVariable(hdf, "Column_Optical_Depth_Aerosols_532")
rh = readVariable(hdf, "Relative_Humidity")
hdf.end()


base = baseInit - np.tile(DEM, baseInit.shape[1]).reshape((baseInit.shape[1],-1)).T
top = topInit - np.tile(DEM, baseInit.shape[1]).reshape((baseInit.shape[1],-1)).T

# filtre altitudes negatives
indAlt = np.where(base<0)
base[indAlt] = np.nan
top[indAlt] = np.nan
aod[indAlt] = np.nan
caod[indAlt[0]] = np.nan

# filtre CAD
indCAD = np.where((cad >-20) & (cad < cadMin))
base[indCAD] = np.nan
top[indCAD] = np.nan
aod[indCAD] = np.nan
caod[indCAD[0]] = np.nan

# filtre ExtinctionQC_532
indQC = np.where(qc>2)
base[indQC] = np.nan
top[indQC] = np.nan
aod[indQC] = np.nan
#caod[indQC[0]] = np.nan

# AOD Uncertainty
indAOD_U = np.where(aodUnc > 99)
base[indAOD_U] = np.nan
top[indAOD_U] = np.nan
aod[indAOD_U] = np.nan
#caod[indAOD_U[0]] = np.nan


### Dust
##########
# extraction data initiales
maskDust = feature == types['dust']
baseDust = np.ma.array(data=base, mask=~maskDust, fill_value=np.nan)
topDust = np.ma.array(data=top, mask=~maskDust, fill_value=np.nan)
# extraction datas layer extended
maskExtDust = featureExt == types['dust']
baseExtDust = np.ma.array(data=base, mask=~maskExtDust, fill_value=np.nan)
topExtDust = np.ma.array(data=top, mask=~maskExtDust, fill_value=np.nan)
## masques combines
maskDE = np.where(maskDust | maskExtDust)[0]
baseDE = np.ma.array(data=base, mask=~maskDE, fill_value=np.nan)
## extraction de la premiere couche
#indDust = np.where(maskDust | maskExtDust)[0]
#listMat = np.vsplit(base[indDust,:], base[indDust,:].shape[0])  # decoupage de la matrice 1D en une liste de n sous-matrices de dim (1,8)
#output = map(indiceCouche1, listMat)
#indices = [m[1] for m in output]
listMat = np.vsplit(base, base.shape[0])  # decoupage de la matrice 1D en une liste de n sous-matrices de dim (1,8)
output = map(indiceCouche1, listMat)
indices = [m[1] for m in output]

#dust = pd.DataFrame()
#dust['lat'] = lat[indDust]
#dust['lon'] = lon[indDust]
#dust['base'] = np.array([m[0] for m in output])
#dust['top'] = np.array([top[indDust[i], indices[i]] for i in range(len(indices))])
#dust['aod'] = np.array([aod[indDust[i], indices[i]] for i in range(len(indices))])
#dust['caod'] = caod[indDust]
#dust['masse_aod'] = 1000 * (dust.aod / (dust.top - dust.base))
#dust['masse_caod'] = 1000 * (dust.caod / (dust.top - dust.base))
#dust.dropna(subset=['base'], inplace=True)

df = pd.DataFrame()
df['lat'] = lat
df['lon'] = lon
df['feature'] = np.array(feature[range(len(indices)), indices])
df['featureExt'] = np.array(featureExt[range(len(indices)), indices])
df['base'] = np.array([m[0] for m in output])
df['top'] = np.array(top[range(len(indices)), indices])
df['aod'] = np.array(aod[range(len(indices)), indices])
df['caod'] = caod
df['masse_aod'] = 1000 * (df.aod / (df.top - df.base))
df['masse_caod'] = 1000 * (df.caod / (df.top - df.base))
df.dropna(subset=['base'], inplace=True)
dust = df[(df.feature==types['dust'])|(df.featureExt==types['dust'])]
pdust = df[(df.feature==types['polluted_dust'])|(df.featureExt==types['polluted_dust'])]

# lissage base
dustF = pd.DataFrame()
dustF['base'], ixs = medFilt(dust.base.values, 9)
dustF['top'] = dust.top.values[ixs]
dustF['aod'] = dust.aod.values[ixs]
dustF['masse_aod'] = dust.masse_aod.values[ixs]
dustF['masse_caod'] = dust.masse_caod.values[ixs]

### Polluted Dust
#################
# extraction data initiales
maskPdust = feature == types['polluted_dust']
basePdust = np.ma.array(data=base, mask=~maskPdust, fill_value=np.nan)
topPdust = np.ma.array(data=top, mask=~maskPdust, fill_value=np.nan)
# extraction datas layer extended
maskExtPdust = featureExt == types['polluted_dust']
baseExtPdust = np.ma.array(data=base, mask=~maskExtPdust, fill_value=np.nan)
topExtPdust = np.ma.array(data=top, mask=~maskExtPdust, fill_value=np.nan)
# masques combines
#maskP = ~maskPdust & ~maskExtPdust
#basePdust1 = np.ma.array(data=base, mask=maskP, fill_value=np.nan)
#topPdust1 = np.ma.array(data=top, mask=maskP, fill_value=np.nan)
#aodPdust1 = np.ma.array(data=aod, mask=maskP, fill_value=np.nan)
# extraction de la premiere couche
#indPdust = np.where(maskPdust | maskExtPdust)[0]
#listMat1 = np.vsplit(base[indPdust,:], base[indPdust,:].shape[0])  # decoupage de la matrice 1D en une liste de n sous-matrices de dim (1,8)
#output1 = map(indiceCouche1, listMat1)
#indices1 = [m[1] for m in output1]
#pdust = pd.DataFrame()
#pdust['lat'] = lat[indPdust]
#pdust['lon'] = lon[indPdust]
#pdust['base'] = np.array([m[0] for m in output1])
#pdust.base.replace(-9999.0, np.nan, inplace=True)
#pdust['top'] = np.array([top[indPdust[i], indices1[i]] for i in range(len(indPdust))])
#pdust['aod'] = np.array([aod[indPdust[i], indices1[i]] for i in range(len(indPdust))])
#pdust['caod'] = caod[indPdust]
#pdust['masse_aod'] = 1000 * (pdust.aod / (pdust.top - pdust.base))
#pdust['masse_caod'] = 1000 * (pdust.caod / (pdust.top - pdust.base))
#pdust.dropna(subset=['base'], inplace=True)

# lissage base
pdustF = pd.DataFrame()
pdustF['base'], ixs1 = medFilt(pdust.base.values, 9)
pdustF['top'] = pdust.top.values[ixs1]
pdustF['aod'] = pdust.aod.values[ixs1]
pdustF['masse_aod'] = pdust.masse_aod.values[ixs1]
pdustF['masse_caod'] = pdust.masse_caod.values[ixs1]


#######################################################################################
fig = plt.figure(1, figsize=(15,10))
fig.suptitle(f.split('/')[-1] + '\nProfil des Dust et Polluted Dust')
gs = gridspec.GridSpec(7,1, hspace=0.1)
ax1 = plt.subplot(gs[:3, :])
ax2 = plt.subplot(gs[4:, :], sharex=ax1)
ax3 = plt.subplot(gs[3], sharex=ax1)
for i in range(8):
    ax1.fill_between(lat, baseDust[:,i], topDust[:,i], color="green", alpha=0.5)
    ax1.fill_between(lat, baseExtDust[:,i], topExtDust[:,i], color="none",hatch="X",edgecolor="blue", alpha=0.5) 
    ax1.xaxis.grid(True)
    ax1.tick_params(axis='x', which='both', top='on', bottom='off', labeltop='on', labelbottom='off')
    ax1.set_ylabel('Altitude(km)')
    ax1.text(0.7, 0.92, 'Dust', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='green', bbox=dict(facecolor='none', edgecolor='green', pad=10.0))
    ax1.text(0.7, 0.77, 'Dust Layer Base Extended', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='blue', bbox=dict(facecolor='none', hatch="X", edgecolor='blue', pad=10.0))
    ax2.fill_between(lat, basePdust[:,i], topPdust[:,i], color="green", alpha=0.5)
    ax2.fill_between(lat, baseExtPdust[:,i], topExtPdust[:,i], color="none", hatch="X", edgecolor='blue', alpha=0.5)
    ax2.set_xlabel('Latitude')
    ax2.text(0.7, -0.45, 'Polluted Dust', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='green', bbox=dict(facecolor='none', edgecolor='green', pad=10.0))
    ax2.text(0.7, -0.60, 'Polluted Dust Layer Base Extended', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='blue', bbox=dict(facecolor='none', hatch="X", edgecolor='blue', pad=10.0))
    ax2.xaxis.grid(True)
    ax2.set_ylabel('Altitude(km)')
    ax3.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
    ax3.plot(lat, np.ma.array(nbLayers, mask=nbLayers==0), color='black', marker='.', ls='none', markeredgecolor='none', markerfacecolor='black')
    ax3.text(0.7, -0.1, 'Nombre de couches detectees', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='black', bbox=dict(facecolor='none', edgecolor='black', pad=10.0))
    ax3.xaxis.grid(True)
    ax3.set_ylabel('nb Layers')
    ax3.set_yticklabels([1,'',2,'',3,'',4])
fig.show()
plt.show()



fig = plt.figure(1, figsize=(15,10))
fig.suptitle(f.split('/')[-1] + '\nProfil des Dust et Polluted Dust')
gs = gridspec.GridSpec(7,1, hspace=0.1)
ax1 = plt.subplot(gs[:3, :])
ax2 = plt.subplot(gs[4:, :], sharex=ax1)
ax3 = plt.subplot(gs[3], sharex=ax1)
for i in range(8):
    ax1.fill_between(lat, baseDust[:,i], topDust[:,i], color="green", alpha=0.5)
    ax1.fill_between(lat, baseExtDust[:,i], topExtDust[:,i], color="green", alpha=0.5) 
    ax1.xaxis.grid(True)
    ax1.tick_params(axis='x', which='both', top='on', bottom='off', labeltop='on', labelbottom='off')
    ax1.set_xlabel('Latitude')
    ax1.set_ylabel('Altitude(km)')
    ax1.text(0.7, 0.92, 'Dust', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='green', bbox=dict(facecolor='none', edgecolor='green', pad=10.0))
    ax2.fill_between(lat, basePdust[:,i], topPdust[:,i], color="green", alpha=0.5)
    ax2.fill_between(lat, baseExtPdust[:,i], topExtPdust[:,i], color="green", alpha=0.5)
    ax2.set_xlabel('Latitude')
    ax2.text(0.7, -0.45, 'Polluted Dust', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='green', bbox=dict(facecolor='none', edgecolor='green', pad=10.0))
    ax2.xaxis.grid(True)
    ax2.set_ylabel('Altitude(km)')
    ax3.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
    ax3.plot(lat, np.ma.array(nbLayers, mask=nbLayers==0), color='black', marker='.', ls='none', markeredgecolor='none', markerfacecolor='black')
    ax3.text(0.7, -0.1, 'Nombre de couches detectees', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='black', bbox=dict(facecolor='none', edgecolor='black', pad=10.0))
    ax3.xaxis.grid(True)
    ax3.set_ylabel('nb Layers')
    ax3.set_yticklabels([1,'',2,'',3,'',4])
ax1.plot(dust.lat, dustF.base, 'r-')
ax1.plot(dust.lat, dust.base, 'r:', marker='o', markeredgecolor='r', markerfacecolor='none')
ax1.plot(dust.lat, dustF.top, 'k-')
ax1.plot(dust.lat, dust.top, 'k:', marker='o', markeredgecolor='k', markerfacecolor='none')
ax2.plot(pdust.lat, pdustF.base, 'r-')
ax2.plot(pdust.lat, pdust.base, 'r:', marker='o', markeredgecolor='r', markerfacecolor='none')
ax2.plot(pdust.lat, pdustF.top, 'k-')
ax2.plot(pdust.lat, pdust.top, 'k:', marker='o', markeredgecolor='k', markerfacecolor='none')
fig.show()
plt.show()



#fig = plt.figure(1, figsize=(17,12))
#fig.suptitle(f)
#gs = gridspec.GridSpec(8,1, hspace=0.05)
#ax1 = plt.subplot(gs[:4, :])
#ax2 = plt.subplot(gs[4:], sharex=ax1)
#for i in range(8):
#    ax1.fill_between(range(4224), baseDust1[:,i], topDust1[:,i], color="green", alpha=0.5)
#    ax1.fill_between(range(4224), baseDust[:,i], topDust[:,i], color="none", hatch="X",edgecolor="blue",alpha=0.5)
#    ax1.fill_between(range(4224), baseExtDust[:,i], topExtDust[:,i], color="none",hatch="X",edgecolor="blue", alpha=0.5) 
#    #ax1.fill_between(lat, baseExtDust[:,i], topExtDust[:,i], color="green", alpha=0.5)
#    #ax1.fill_between(lat, baseDust1[:,i], topDust1[:,i], color="blue", alpha=0.5)
#    ax1.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
#    ax1.xaxis.grid(True)
#    ax1.set_ylabel('Altitude(km)')
#    #ax2.plot(lat, nbLayers, color='red', marker='.', ls='none', markeredgecolor='red', markerfacecolor='none')
#    #ax2.fill_between(lat, baseDust1[:,i], topDust1[:,i], color="blue", alpha=0.5)
#    ax2.fill_between(range(4224), basePdust1[:,i], topPdust1[:,i], color="green", alpha=0.5)
#    ax2.fill_between(range(4224), basePdust[:,i], topPdust[:,i], color="none", hatch="X", edgecolor='blue', alpha=0.5)
#    ax2.fill_between(range(4224), baseExtPdust[:,i], topExtPdust[:,i], color="none", hatch="X", edgecolor='blue', alpha=0.5)
#    ax2.xaxis.grid(True)
#    ax2.grid(True,'minor',color='k', alpha=0.2, ls='-', lw=0.2)
#    ax2.set_xlabel('Latitude')
#    ax2.set_ylabel('nb Layers')
#fig.show()
#plt.show()
#
#
fig = plt.figure(1, figsize=(17,12))
fig.suptitle(f)
gs = gridspec.GridSpec(7,1, hspace=0.05)
ax1 = plt.subplot(gs[:6, :])
ax2 = plt.subplot(gs[6], sharex=ax1)
for i in range(8):
    ax1.fill_between(lat, baseDust[:,i], topDust[:,i], color="k", alpha=0.5)
    #ax1.fill_between(lat, baseExtDust[:,i], topExtDust[:,i], color="green", alpha=0.5)
    #ax1.fill_between(lat, baseDust1[:,i], topDust1[:,i], color="blue", alpha=0.5)
    ax1.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
    ax1.xaxis.grid(True)
    ax1.set_ylabel('Altitude(km)')
    ax1.text(45, 38, 'dust', style='italic',bbox={'facecolor':'green', 'alpha':0.5, 'pad':10})
    ax1.text(45, 35, 'polluted dust', style='italic',bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})

    ax2.plot(lat, nbLayers, color='red', marker='.', ls='none', markeredgecolor='red', markerfacecolor='none')
    ax2.xaxis.grid(True)
    ax2.set_xlabel('Latitude')
    ax2.set_ylabel('nb Layers')
    ax2.set_yticklabels([1,'',2,'',3,'',4])
fig.show()