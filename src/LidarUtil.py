# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import median_filter


####################################################################################
####################################################################################


def calcul_Nbpoints(matrice, point):
    """
    Definit le nombre de points lidar dans un intervalle de +- 0.5 degre de latitude
    
    PARAMETRES
    
    **matrice**(*1D array*): ensemble des latitudes \n
    **point**: latitude \n
    """
    latmin =  point - 0.5
    latmax = point + 0.5
    if matrice[0] < matrice[-1]:
        if latmin < matrice[0]:
            latmin = matrice[0]
        if latmax > matrice[-1]:
            latmax = matrice[-1]
    else:
        if latmin < matrice[-1]:
            latmin = matrice[-1]
        if latmax > matrice[0]:
            latmax = matrice[0]
    ind = np.where( (matrice >= latmin) & (matrice <= latmax) )[0]  # extraction de la liste des indices
    return ind.shape[0]


####################################################################################
####################################################################################


def indiceCouche(matrice):
    """
    retourne la valeur et l'indice de la 1ere couche ou -9999, -9999 si aucune couche n'est valide

    Parametres:
    **matrice (*2d array*)
    
    """
    ind = np.where(matrice.flatten()[:] != -9999)[0]
    if ind.size:
        return matrice.flatten()[ind[-1]],ind[-1]
    else:
        return -9999,-9999
    
    
####################################################################################
####################################################################################   


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

####################################################################################
####################################################################################   


def decodeIGBP(indice):
    """
    Conversion indice IGBP --> nom
    """
    IGBPcode = ['Evergreen_Needleleaf_Forest', 'Evergreen_Broadleaf_Forest',
                'Deciduous_Needleleaf_Forest', 'Deciduous_Broadleaf_Forest',
                'Mixed_Forest', 'Closed_Shrublands', 'Open_Shrubland(Desert)',
                'Woody_Savanna', 'Savanna', 'Grassland', 'Wetland', 'Cropland',
                'Urban', 'Crop_Mosaic', 'Permanent_Snow', 'Barren/Desert',
                'Water', 'Tundra']
    return IGBPcode[indice-1]


####################################################################################
####################################################################################


def lissage1(df_in, size, variable):
    """
    PARAMETRES:

    **df_in** (*pandas dataframe*): dataframe \n
    **size** (*int impair*): dimension de la fenetre \n
    **variableslist** (*list*): liste des variables a traiter \n
    **variable** (*string*): variable de reference pour modifier les variables suivantes sur les memes indices

    Renvoie une dataframe avec les memes dimensions

    """
    assert (size % 2 == 1), "La taille de la fenetre doit etre impaire"

    dataframe = df_in.copy()
    dataframe[variable] = median_filter(dataframe[variable].values, size=size)
    nonmodif_idx = np.where(df_in[variable].values == dataframe[variable])[0]  # recherche des indices de valeurs non modifiees
    modif_idx = np.where(df_in[variable].values != dataframe[variable])[0]  # recherche des indices de valeurs modifiees
    for v in list(set(df_in.columns) - set([variable])):
        mat = dataframe[v].values[:]
        mat_out = np.zeros(mat.shape[0])
        mat_out[:] = np.nan
        for idx in modif_idx:
            if idx-((size-1)/2) < 0:
                diff_valeurs = np.abs(idx - (size / 2) )
                mat_tmp = np.append(mat[:idx+(size / 2)+1], mat[-diff_valeurs:])  # rajout de valeurs de fin de matrice en debut pour eviter valeurs nulles
                mat_out[idx] = np.median(mat_tmp)
            elif idx + ((size - 1) / 2) + 1 > mat.shape[0]:
                diff_valeurs = idx+(size/2)+1 - mat.shape[0]
                mat_tmp = np.append(mat[idx - (size / 2):], mat[:diff_valeurs])  # rajout de valeurs de debut de matrice en fin pour eviter valeurs nulles
                mat_out[idx] = np.median(mat_tmp)
            else:
                mat_out[idx] = np.median(mat[idx-(size/2):idx+(size/2)+1])
            mat_out[nonmodif_idx] = mat[nonmodif_idx]
        dataframe[v] = mat_out[:]
    return dataframe

def argMedian(a):
    m = np.median(a)
    ind = np.argsort(a)[(len(a)/2)+1]
    return m, ind


def lissage(matrice, ww, lref):
    arr = matrice.copy()
    lo = range(arr.shape[1])
    lo.remove(lref)
    for i in range(ww/2,arr.shape[0]-(ww/2)):
        med, ix = argMedian(arr[i-((ww-1)/2):i+((ww-1)/2)+1, lref]) # mediane et indice(dans la fenetre de 9 valeurs) de valeur utilisée pour remplacer la valeur cible [i] de la couche de ref
        if arr[i, lref] != med:
            arr[i, lref] = med
            # modif des valeurs des couches suivantes en utilisant les mêmes indices
            arr[i,lo[0]] = arr[i-((ww-1)/2):i+((ww-1)/2)+1,lo[0]][ix] 
            arr[i,lo[1]] = arr[i-((ww-1)/2):i+((ww-1)/2)+1,lo[1]][ix]
            arr[i,lo[2]] = arr[i-((ww-1)/2):i+((ww-1)/2)+1,lo[2]][ix]
    return arr



import pysal as ps
import pandas as pd
'''
Arguments
---------
dbfile  : DBF file - Input to be imported
upper   : Condition - If true, make column heads upper case
'''
def dbf2DF(dbfile, upper=True): #Reads in DBF files and returns Pandas DF
    db = ps.open(dbfile) #Pysal to open DBF
    d = {col: db.by_col(col) for col in db.header} #Convert dbf to dictionary
    #pandasDF = pd.DataFrame(db[:]) #Convert to Pandas DF
    pandasDF = pd.DataFrame(d) #Convert to Pandas DF
    if upper == True: #Make columns uppercase if wanted 
        pandasDF.columns = map(str.upper, db.header) 
    db.close() 
    return pandasDF