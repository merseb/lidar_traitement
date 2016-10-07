# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.ndimage import median_filter
import sys, os
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt

path = os.path.expanduser('~')+"/code/python/lidar_traitement/src"
sys.path.append(path)
from rolling_window import *
from LidarUtil import *


# liste des variables/parametres extraits de chaque fichier lidar
varlist = ["IGBP_Surface_Type", "Day_Night_Flag", "DEM_Surface_Elevation", "Column_Optical_Depth_Aerosols_532", "Feature_Optical_Depth_532","Feature_Optical_Depth_Uncertainty_532", "ExtinctionQC_532", "CAD_Score", "Feature_Classification_Flags", "Number_Layers_Found","Layer_Base_Extended", "Relative_Humidity", "Layer_Top_Altitude", "Layer_Base_Altitude"]

# Variables/parametres sur lesquelles lissees
param_lissage = ['Column_Optical_Depth_Aerosols_532', 'Base_corr', 'Top_corr']   ##, 'Feature_Optical_Depth_532', 'Feature_Optical_Depth_Uncertainty_532', 'ExtinctionQC_532', 'Relative_Humidity']



def lissage(df_in, size, variable):
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


ddir = '/home/mers/code/python/lidar_traitement'
os.chdir(ddir)
f = 'zone_etude/CAL_LID_L2_05kmALay-Prov-V3-30.2014-03-24T13-45-37ZD.hdf'
hdf = SD(f, SDC.READ)
df_file = pd.DataFrame()
df_file["Latitude"] = hdf.select('Latitude')[:, 1]
df_file["Longitude"] = hdf.select('Longitude')[:, 1]

##### Extraction de l'indice de la 1ere couche valide en s appuyant sur la 1ere variable definie dans layers_ref
base = hdf.select('Layer_Base_Altitude')
base_mat = base[:]
for i in range(8):
    df_file[str(i)]=np.nan
df_file.loc[df_file.index, [str(i) for i in range(8)]] = base_mat[:]
list_mat_in = np.vsplit(base_mat, base_mat.shape[0])  # decoupage de la matrice 1D en une liste de n sous-matrices de dim (1,8)
list_mat_out = map(indiceCouche, list_mat_in)  # appel fonction indiceCouche pour recuperer l'indice de la 1ere couche valide pour chaque coord lat/lon      
df_file['Layer_Base_Altitude'] = [m[0] for m in list_mat_out]
indices = [m[1] for m in list_mat_out]  # liste des indices qui seront ensuite utilisés comme 'masque' pour les autres couches
base.endaccess()
#####

##### boucle d'extraction des valeurs de chaque variable correspondant aux indices definis precedemment
for var in list(set(varlist) - set(['Layer_Base_Altitude'])):  # 1ere variable de layers_ref exclue
    try:
        var_dict = hdf.select(var)
        mat_in = var_dict[:]
        attribs = var_dict.attributes()
        if hdf.datasets()[var][1][1] == 1:
            df_file[var] = mat_in[:].flatten()
            #print var
        elif var == 'DEM_Surface_Elevation':
            df_file[var] = mat_in[:, 2]  # variable DEM: utilisation de la valeur moyenne
        else:
            val = []
            print var
            for i in range(len(indices)):
                if indices[i] != -9999:
                    val.append(mat_in[i, indices[i]])
                else:
                    val.append(-9999)
            df_file[var] = val
    except KeyError:
        df_file[var] = np.nan
    var_dict.endaccess()
hdf.end()
df_file = df_file[df_file.Layer_Base_Altitude != -9999]
##### suppression des valeurs pour lesquelles la couche basse est détectée sous l'altitude du DEM ou au-dessus la couche top
df_file['tmp'] = df_file.Layer_Base_Altitude - df_file.DEM_Surface_Elevation
df_file['tmp1'] = df_file.Layer_Top_Altitude - df_file.Layer_Base_Altitude
df_file = df_file[(df_file.tmp > 0) & (df_file.tmp1 > 0)]
df_file.drop(['tmp','tmp1'], axis=1, inplace=True)


##### calcul de Layer_Top_Altitude et Layer_Base_Altitude corrigées a partir du DEM
df_file['Top_corr'] = df_file.Layer_Top_Altitude - df_file.DEM_Surface_Elevation
df_file['Base_corr'] = df_file.Layer_Base_Altitude - df_file.DEM_Surface_Elevation
df_file.drop(['Layer_Base_Altitude', 'Layer_Top_Altitude'], axis=1, inplace=True)
#####
dfFiltre = df_file[(df_file.CAD_Score < -20) & ((df_file.ExtinctionQC_532 == 0) | (df_file.ExtinctionQC_532 == 1)) & (df_file.Feature_Optical_Depth_Uncertainty_532 < 99) & ((df_file.FeatureSubtype == 'dust') | (df_file.FeatureSubtype == 'polluted_dust'))]

##### Conversion int16 en sous-categories
df_file['FeatureSubtype'] = df_file.Feature_Classification_Flags.apply(decodeFeatureMask)
#####

##### conversion code IGBP denomination 
df_file.IGBP_Surface_Type = df_file.IGBP_Surface_Type.apply(decodeIGBP)
#####


dfFiltre = df_file[(df_file.CAD_Score < -20) & ((df_file.ExtinctionQC_532 == 0) | (df_file.ExtinctionQC_532 == 1)) & (df_file.Feature_Optical_Depth_Uncertainty_532 < 99) & ((df_file.FeatureSubtype == 'dust') | (df_file.FeatureSubtype == 'polluted_dust'))]


params = [p+'_lissage' for p in param_lissage]
for c in params:
    df[c] = np.nan

w_lissage = 25
for subtype in ['dust', 'polluted_dust']:
    tmp = lissage(df[df.FeatureSubtype == subtype][param_lissage], w_lissage, 'Base_corr')
    df.loc[df[df.FeatureSubtype == subtype].index, params] = tmp.values



df = df_file[(df_file.Latitude > 8) & (df_file.Latitude < 22)]

#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax2 = ax1.twiny()
#ax1.plot(df.Latitude,df.Base_corr, 'k o', label='base')
#ax1.plot(df.Latitude,df.Top_corr, 'r o', label='top')
##ax2.plot(df.Latitude,df.Column_Optical_Depth_Aerosols_532, 'g-', label='AOD')
#h1, l1 = ax1.get_legend_handles_labels()
#h2, l2 = ax2.get_legend_handles_labels()
##ax1.legend(h1+h2, l1+l2, loc='upper right')
#fig.title('Base et Top Layer 2014-03-24 ')
#fig.show()
m_dust = df.mask(df.FeatureSubtype == 'dust')
m_pdust = df.mask(df.FeatureSubtype == 'polluted_dust')
plt.plot(df.Latitude,m_pdust.Base_corr, 'k o', label='base dust')
plt.plot(df.Latitude,m_dust.Base_corr, 'k *', label='base polluted dust')
plt.plot(df.Latitude,m_pdust.Top_corr, 'r o', label='top dust')
plt.plot(df.Latitude,m_dust.Top_corr, 'r *', label='top polluted dust')
plt.legend(), plt.title('Base et Top Layer 2014-03-24 dust et polluted dust confondus')
plt.gca().yaxis.grid(False)
plt.xlabel('Latitudes')
plt.ylabel('Altitude (km)')
plt.show()





plt.plot(df.Latitude,df.Base_corr, linestyle='-', marker='o', color='b', label='base dust + polluted dust')
plt.plot(df.Latitude,df.Base_corr_lissage, linestyle='--', marker='*', color='k', label='base dust + polluted dust lissage 25val')
plt.plot(df.Latitude,df.Top_corr, linestyle='-', marker='o', color='r', label='top dust + polluted dust')
plt.plot(df.Latitude,df.Top_corr_lissage, linestyle='--', marker='*', color='g', label='top dust + polluted dust lissage 25val')
plt.title('Base et Top Layer 2014-03-24 dust et polluted dust confondus:\ncomparaison lissage 25 valeurs')
plt.legend(), plt.show()

plt.plot(df.Latitude,df.Column_Optical_Depth_Aerosols_532, linestyle='-', marker='o', color='b', label='AOD dust + polluted dust')
plt.plot(df.Latitude,df.Column_Optical_Depth_Aerosols_532_lissage, linestyle='--', marker='*', color='k', label='AOD dust + polluted dust lissage 25val')
plt.title('Column_Optical_Depth_Aerosols_532 2014-03-24 dust et polluted dust confondus:\ncomparaison lissage 25 valeurs')
plt.legend(), plt.show()

lats = []
for i in range(df_file.Latitude.min().round(), df_file.Latitude.max().round(), 1):
    lats.append(df_file[(df_file.Latitude >= i) & (df_file.Latitude < (i + 1))])

try:
    for subtype in subtypes:
        ##### lissage (mediane) n fois en fct du nombre de variables choisies(layers_ref)
        for lref in layers_ref[:]:           
            if lref == 'Base_corr':
                tmp = lissage(df_file[df_file.FeatureSubtype == subtype][param_lissage], w_lissage, lref)
                df_file.loc[df_file[df_file.FeatureSubtype == subtype].index, params] = tmp.values
            else:
                lrf = lref + '_lissage'
                tmp = lissage(df_file[df_file.FeatureSubtype == subtype][params], w_lissage, lrf)
                df_file.loc[df_file[df_file.FeatureSubtype == subtype].index, params] = tmp.values
except:
    pass
df1 = df_file