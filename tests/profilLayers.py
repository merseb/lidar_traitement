# -*- coding: utf-8 -*-

import numpy as np
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
from rolling_window import *
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

try:
    f = sys.argv[1]
except IndexError:
    print 'saisir un chemin de fichier'
    sys.exit()
try:
    features = sys.argv[2]
except IndexError:
    features = ''    
if features not in ['all','']:
    print 'saisir argument all ou laisser vide'
    sys.exit()
#f = path + '/2014/zone_etude/CAL_LID_L2_05kmALay-Prov-V3-30.2014-03-24T13-45-37ZD.hdf'
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
hdf.end()


base = baseInit - np.tile(DEM, baseInit.shape[1]).reshape((baseInit.shape[1],-1)).T
top = topInit - np.tile(DEM, baseInit.shape[1]).reshape((baseInit.shape[1],-1)).T
# filtre altitudes negatives
indAlt = np.where(base<0)
base[indAlt] = np.nan
top[indAlt] = np.nan


# filtre CAD
indCAD = np.where((cad >-20) & (cad < cadMin))
base[indCAD] = np.nan
top[indCAD] = np.nan


# filtre ExtinctionQC_532
indQC = np.where(qc>2)
base[indQC] = np.nan
top[indQC] = np.nan

# AOD Uncertainty
indAOD_U = np.where(aodUnc > 99)
base[indAOD_U] = np.nan
top[indAOD_U] = np.nan


### Dust
# extraction data initiales
maskDust = feature == types['dust']
baseDust = np.ma.array(data=base, mask=~maskDust, fill_value=np.nan)
topDust = np.ma.array(data=top, mask=~maskDust, fill_value=np.nan)
# extraction datas layer extended
maskExtDust = featureExt == types['dust']
baseExtDust = np.ma.array(data=base, mask=~maskExtDust, fill_value=np.nan)
topExtDust = np.ma.array(data=top, mask=~maskExtDust, fill_value=np.nan)
# masques combines
maskD_IE = ~maskDust & ~maskExtDust
baseDustIE = np.ma.array(data=base, mask=maskD_IE, fill_value=np.nan)
topDustIE = np.ma.array(data=top, mask=maskD_IE, fill_value=np.nan)
# extraction de la premiere couche
listMat = np.vsplit(base, base.shape[0])  # decoupage de la matrice 1D en une liste de n sous-matrices de dim (1,8)
output = map(indiceCouche1, listMat)
indices = [m[1] for m in output]
base0 = np.array([m[0] for m in output])
top0 = np.array(top[range(top.shape[0]), indices])
feature0 = np.array(feature[range(top.shape[0]), indices])
featureExt0 = np.array(featureExt[range(top.shape[0]), indices])

ind=np.where((featureExt0!=2) & (featureExt0!=5) & (feature0!=2) & (feature0!=5))[0]
base0[ind] = np.nan
top0[ind] = np.nan


### Polluted Dust
# extraction data initiales
maskPdust = feature == types['polluted_dust']
basePdust = np.ma.array(data=base, mask=~maskPdust, fill_value=np.nan)
topPdust = np.ma.array(data=top, mask=~maskPdust, fill_value=np.nan)
# extraction datas layer extended
maskExtPdust = featureExt == types['polluted_dust']
baseExtPdust = np.ma.array(data=base, mask=~maskExtPdust, fill_value=np.nan)
topExtPdust = np.ma.array(data=top, mask=~maskExtPdust, fill_value=np.nan)
# masques combines
maskP_IE = ~maskPdust & ~maskExtPdust
basePdustIE = np.ma.array(data=base, mask=maskP_IE, fill_value=np.nan)
topPdustIE = np.ma.array(data=top, mask=maskP_IE, fill_value=np.nan)


### Clean Marine
maskCmarine = feature == types['clean_marine']
# extraction datas layer extended
maskExtCmarine = featureExt == types['clean_marine']
# masques combines
maskCM_IE = ~maskCmarine & ~maskExtCmarine
baseCmarineIE = np.ma.array(data=base, mask=maskCM_IE, fill_value=np.nan)
topCmarineIE = np.ma.array(data=top, mask=maskCM_IE, fill_value=np.nan)


### "polluted_continental
maskPcont = feature == types['polluted_continental']
# extraction datas layer extended
maskExtPcont = featureExt == types['polluted_continental']
# masques combines
maskPC_IE = ~maskPcont & ~maskExtPcont
basePcontIE = np.ma.array(data=base, mask=maskPC_IE, fill_value=np.nan)
topPcontIE = np.ma.array(data=top, mask=maskPC_IE, fill_value=np.nan)

### clean_continental
maskCcont = feature == types['clean_continental']
# extraction datas layer extended
maskExtCcont = featureExt == types['clean_continental']
# masques combines
maskCC_IE = ~maskCcont & ~maskExtCcont
baseCcontIE = np.ma.array(data=base, mask=maskCC_IE, fill_value=np.nan)
topCcontIE = np.ma.array(data=top, mask=maskCC_IE, fill_value=np.nan)

###  smoke
maskS = feature == types['smoke']
# extraction datas layer extended
maskExtS = featureExt == types['smoke']
# masques combines
maskS_IE = ~maskS& ~maskExtS
baseSmokeIE = np.ma.array(data=base, mask=maskS_IE, fill_value=np.nan)
topSmokeIE = np.ma.array(data=top, mask=maskS_IE, fill_value=np.nan)


#######################################################################################
#fig = plt.figure(1, figsize=(15,10))
#fig.suptitle(f.split('/')[-1] + '\nProfil des Dust et Polluted Dust')
#gs = gridspec.GridSpec(7,1, hspace=0.1)
#ax1 = plt.subplot(gs[:3, :])
#ax2 = plt.subplot(gs[4:, :], sharex=ax1)
#ax3 = plt.subplot(gs[3], sharex=ax1)
#for i in range(8):
#    ax1.fill_between(lat, baseDust[:,i], topDust[:,i], color="green", alpha=0.5)
#    ax1.fill_between(lat, baseExtDust[:,i], topExtDust[:,i], color="none",hatch="X",edgecolor="blue", alpha=0.5) 
#    ax1.xaxis.grid(True)
#    ax1.tick_params(axis='x', which='both', top='on', bottom='off', labeltop='on', labelbottom='off')
#    ax1.set_ylabel('Altitude(km)')
#    ax1.text(0.7, 0.92, 'Dust', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='green', bbox=dict(facecolor='none', edgecolor='green', pad=10.0))
#    ax1.text(0.7, 0.77, 'Dust Layer Base Extended', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='blue', bbox=dict(facecolor='none', hatch="X", edgecolor='blue', pad=10.0))
#    ax2.set_xlabel('Latitude')
#    ax2.text(0.7, -0.45, 'Polluted Dust', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='green', bbox=dict(facecolor='none', edgecolor='green', pad=10.0))
#    ax2.text(0.7, -0.60, 'Polluted Dust Layer Base Extended', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='blue', bbox=dict(facecolor='none', hatch="X", edgecolor='blue', pad=10.0))
#    ax2.fill_between(lat, basePdust[:,i], topPdust[:,i], color="green", alpha=0.5)
#    ax2.fill_between(lat, baseExtPdust[:,i], topExtPdust[:,i], color="none", hatch="X", edgecolor='blue', alpha=0.5)
#    ax2.xaxis.grid(True)
#    ax3.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
#    ax2.set_ylabel('Altitude(km)')
#    ax3.plot(lat, np.ma.array(nbLayers, mask=nbLayers==0), color='black', marker='.', ls='none', markeredgecolor='none', markerfacecolor='black')
#    ax3.text(0.7, -0.1, 'Nombre de couches detectees', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='black', bbox=dict(facecolor='none', edgecolor='black', pad=10.0))
#    ax3.xaxis.grid(True)
#    ax3.set_ylabel('nb Layers')
#    ax3.set_yticklabels([1,'',2,'',3,'',4])
#fig.show()
#plt.show()


fig = plt.figure(1, figsize=(17,12))
title = f + '\nDust et Polluted Dust couches initiales et extended\n2: Dust, 5: Polluted Dust'
gs = gridspec.GridSpec(8,1, hspace=0.05)
ax1 = plt.subplot(gs[:7, :])
ax2 = plt.subplot(gs[7:], sharex=ax1)
for i in range(8):
    ax1.fill_between(lat, baseDustIE[:,i], topDustIE[:,i], color="yellow", alpha=0.5)
    ax1.fill_between(lat, basePdustIE[:,i], topPdustIE[:,i], color="brown", alpha=0.5)
    if features == 'all':
        title = f + '\nDust et Polluted Dust couches initiales et extended\n1: Clean Marine, 2: Dust, 3:Polluted Continental, 4: Clean Continental, 5: Polluted Dust, 6: Smoke'
        ax1.fill_between(lat, baseCmarineIE[:,i], topCmarineIE[:,i], color="blue", alpha=0.5)
        ax1.fill_between(lat, basePcontIE[:,i], topPcontIE[:,i], color="red", alpha=0.5)
        ax1.fill_between(lat, baseCcontIE[:,i], topCcontIE[:,i], color="green", alpha=0.5)
        ax1.fill_between(lat, baseSmokeIE[:,i], topSmokeIE[:,i], color="grey", alpha=0.5)
        ax1.text(0.9, 0.80, '1', horizontalalignment='right', verticalalignment='top', transform=ax1.transAxes, color='black', bbox=dict(facecolor='blue', edgecolor='blue', pad=10.0))
        ax1.text(0.9, 0.74, '3', horizontalalignment='right', verticalalignment='top', transform=ax1.transAxes, color='black', bbox=dict(facecolor='red', edgecolor='red', pad=10.0))
        ax1.text(0.9, 0.68, '4', horizontalalignment='right', verticalalignment='top', transform=ax1.transAxes, color='black', bbox=dict(facecolor='green', edgecolor='green', pad=10.0))
        ax1.text(0.9, 0.62, '6', horizontalalignment='right', verticalalignment='top', transform=ax1.transAxes, color='black', bbox=dict(facecolor='grey', edgecolor='grey', pad=10.0))
    ax1.tick_params(axis='x', which='both', top='on', bottom='off', labelbottom='off', labeltop='on')
    ax1.xaxis.grid(True)
    ax1.set_ylabel('Altitude(km)')
    ax1.text(0.9, 0.92, '2', horizontalalignment='right', verticalalignment='top', transform=ax1.transAxes, color='black', bbox=dict(facecolor='yellow', edgecolor='yellow', pad=10.0))
    ax1.text(0.9, 0.86, '5', horizontalalignment='right', verticalalignment='top', transform=ax1.transAxes, color='black', bbox=dict(facecolor='brown', edgecolor='brown', pad=10.0))
    ax2.plot(lat, np.ma.array(nbLayers, mask=nbLayers==0), color='black', marker='.', ls='none', markeredgecolor='none', markerfacecolor='black')
    ax2.text(0.9, -0.05, 'Nombre de couches detectees', horizontalalignment='right', verticalalignment='top', transform=ax1.transAxes, color='black', bbox=dict(facecolor='none', edgecolor='black', pad=10.0))
    ax2.xaxis.grid(True)
    ax2.set_xlabel('Latitude')
    ax2.tick_params(axis='x', which='both', top='off', bottom='on', labelbottom='on')
    ax2.set_ylabel('nb Layers')
    ax2.set_yticklabels([1,'',2,'',3,'',4])
ax1.plot(lat, base0, 'k-', label='limites base-top 1ere couche detectee')
ax1.plot(lat, top0, 'k-')
ax1.legend()
fig.suptitle(title)
fig.show()
plt.show()

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