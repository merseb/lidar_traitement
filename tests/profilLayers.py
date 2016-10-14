# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os, sys
from glob import glob
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ddir_fig =os.path.expanduser('~') +'/Bureau/teledm/fusion_donnees/resultats/figures'
path = os.path.expanduser('~')+"/code/python/lidar_traitement"
sys.path.append(path+"/src")
from rolling_window import *
from LidarUtil import *
os.chdir(path+'/zone_etude')

# liste des variables/parametres extraits de chaque fichier lidar
varlist = ["IGBP_Surface_Type", "Day_Night_Flag", "DEM_Surface_Elevation", "Column_Optical_Depth_Aerosols_532", "Feature_Optical_Depth_532","Feature_Optical_Depth_Uncertainty_532", "ExtinctionQC_532", "CAD_Score", "Feature_Classification_Flags", "Number_Layers_Found","Layer_Base_Extended", "Relative_Humidity", "Layer_Top_Altitude", "Layer_Base_Altitude"]

# Variables/parametres sur lesquelles lissees
varLissage = ['Base_corr', 'Top_corr', 'Column_Optical_Depth_Aerosols_532',  'Concentration_Aerosols']   ##, 'Feature_Optical_Depth_532', 'Feature_Optical_Depth_Uncertainty_532', 'ExtinctionQC_532', 'Relative_Humidity']
subtypes = ['dust', 'polluted_dust']


def decodeFeatureMask(int16):
    """
    Flag: conversion int16 --> int
    La fonction retourne une matrice de 3 valeurs chacune correspondant aux flags (1,2,3)
    
    Subtype                  
    0 = not determined      
    1 = clean marine   
    2 = pure dust
    3 = polluted continental
    4 = clean continental
    5 = polluted dust
    6 = smoke
    7 = other
    """
    
    
    binaire = format(int16,'016b')  # little endian 
    FeatureType = np.int(binaire[-3:],2)
    FeatureTypeQA = np.int(binaire[-5:-3],2)
    #IceWaterPhase = np.int(binaire[-7:-5],2)
    #IceWaterPhaseQA = np.int(binaire[-9:-7],2)
    FeatureSubtype = np.int(binaire[-12:-9],2)
    #CloudAerosolPSCTypeQA = np.int(binaire[-13],2)
    #Horizonthalaveraging = np.int(binaire[:-13],2)
    #list_feature = [FeatureType,FeatureTypeQA,IceWaterPhase,IceWaterPhaseQA,FeatureSubtype,CloudAerosolPSCTypeQA,Horizonthalaveraging]
    #return [binaire,FeatureSubtype]
    if FeatureType==3 and FeatureTypeQA>1:
        if FeatureSubtype == 0:
            return 0
        elif FeatureSubtype == 1:
            return 1  # "clean_marine"
        elif FeatureSubtype == 2:
            return 2  # "dust"
        elif FeatureSubtype == 3:
            return 3  # "polluted_continental"
        elif FeatureSubtype == 4:
            return 4  # "clean_continental"
        elif FeatureSubtype == 5:
            return 5  # "pollute_dust"
        elif FeatureSubtype == 6:
            return 6  # "smoke"
        elif FeatureSubtype == 7:
            return 0
    else:
        return 0


types = {"clean_marine":1, "dust":2, "polluted_continental":3, "clean_continental":4, "polluted_dust":5, "smoke":6}



f = 'CAL_LID_L2_05kmALay-Prov-V3-30.2014-03-24T13-45-37ZD.hdf'
#f = 'CAL_LID_L2_05kmALay-Prov-V3-30.2014-02-08T00-57-04ZN.hdf'
date = f[31:41]
hdf = SD(f, SDC.READ)
lat = hdf.select('Latitude')[:, 1]
lon = hdf.select('Longitude')[:, 1]
featIn = hdf.select('Feature_Classification_Flags')[:]
feature = np.array([decodeFeatureMask(int16) for int16 in featIn.flatten().tolist()]).reshape(featIn.shape)
mask = feature.copy().astype(float)
mask[mask != types['dust']] = np.nan
base = hdf.select('Layer_Base_Altitude')[:] * mask
top = hdf.select('Layer_Top_Altitude')[:] * mask

fig = plt.figure(figsize=(23,12))
ax = fig.add_subplot(111, projection='3d')
for i in range(8):
    ax.scatter(lon, lat,base[:,i], c='k')
    ax.scatter(lon, lat,top[:,i], c='r')
plt.show()