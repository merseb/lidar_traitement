# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os
from glob import glob



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
            return "undeterminate"
        elif FeatureSubtype == 1:
            return "clean_marine"
        elif FeatureSubtype == 2:
            return "dust"
        elif FeatureSubtype == 3:
            return "polluted_continental"
        elif FeatureSubtype == 4:
            return "clean_continental"
        elif FeatureSubtype == 5:
            return "polluted_dust"
        elif FeatureSubtype == 6:
            return "smoke"
        elif FeatureSubtype == 7:
            return "other"
    else:
        return "no_aerosol"


############ test decodeFeatureMask ##########################################
if __name__ == "__main__":
    
    ddir = '/home/mers/Bureau/teledm/donnees/lidar/calipso/caliop/out'
    os.chdir(ddir)
    files = sorted(glob('*.csv'))
    for f in files[:1]:
        print f
        csv = pd.read_csv(f,header=0)
        #df = pd.DataFrame(csv[['Latitude','Longitude','Layer_Base_Altitude','Layer_Top_Altitude','Feature_Classification_Flags']])
        int16_flags = list(csv.Feature_Classification_Flags.values)
        masks = [decodeFeatureMask(fg) for fg in int16_flags]
        binaires = [m[0] for m in masks]
        df_test = csv.join(pd.DataFrame(np.vstack(masks),columns=['binaire','FeatureType','FeatureTypeQA','FeatureSubtype']))
        df_test[['FeatureType','FeatureTypeQA','FeatureSubtype']] = df_test[['FeatureType','FeatureTypeQA','FeatureSubtype']].astype('int')
        df_test_filter = df_test[(df_test.FeatureType==3) & (df_test.FeatureTypeQA>0)]
        df_test.to_csv('flag_test_'+f[:-3]+'csv', index=False)
        df_test_filter.to_csv('flag_test_filter_'+f[:-3]+'csv', index=False)