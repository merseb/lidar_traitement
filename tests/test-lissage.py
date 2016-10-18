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
varLissage = ['Base_corr', 'Top_corr', 'Column_Optical_Depth_Aerosols_532',  'Concentration_Aerosols']   ##, 'Feature_Optical_Depth_532', 'Feature_Optical_Depth_Uncertainty_532', 'ExtinctionQC_532', 'Relative_Humidity']
subtypes = ['dust', 'polluted_dust']


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


def argMedianLat(a):
    N = (a.shape[0]/2) + 1
    if (a[0,0] < dust[i,0] -0.5) and (a[-1,0] > a[N,0] + 0.5):
        return a[N,2]
    else:
        med[N] = np.median(dust[:, 2])
    return m, ind


def lissage1(latitudes, matrice, ww, lref):
    k = ww//2 ## fenetre / 2 
    array = matrice.copy()
    lo = range(arr[:].shape[1]) ## liste des indices des variables 
    lo.remove(lref)  ## oter l'indice de la variable cible
    for i in range(k,array.shape[0]-k):  ## boucle de la fenetre glissante
        arrTmp = array[i-k:i+k+1, lref]  ## extraction des valeurs de la fenetre pour la variable cible
        lats = latitudes[i-k:i+k+1]  ## extraction des latitudes correspondantes
        if (lats[0] < arrTmp[k+1]-0.5) and (lats[-1] < arrTmp[k+1]+0.5):  ## condition: la dif de latitude dans la fenetre ne doit pas etre > a 1 degre
            med = np.median(arrTmp)  ## calcul de la mediane
            ix = np.argsort(arrTmp)[k+1]  ## calcul de l indice de la valeur mediane
            if array[i, lref] != med:  ## si cette nouvelle valeur est différente de la valeur actuelle de la matrice --> modifier les valeurs pour chaque variable
                array[i, lref] = med  ## modif de la variable cible
                for l in lo:
                    array[i,l] = array[i-k:i+k+1,l][ix] # modif des valeurs des couches suivantes en utilisant le meme indice
        else:
    return array


ddir = '/home/mers/code/python/lidar_traitement/zone_etude'
ddir_fig ='/home/mers/Bureau/teledm/fusion_donnees/resultats/figures'
os.chdir(ddir)
f = 'CAL_LID_L2_05kmALay-Prov-V3-30.2014-03-24T13-45-37ZD.hdf'
#f = 'CAL_LID_L2_05kmALay-Prov-V3-30.2014-02-08T00-57-04ZN.hdf'
date = f[31:41]
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

##### Conversion int16 en sous-categories
df_file['FeatureSubtype'] = df_file.Feature_Classification_Flags.apply(decodeFeatureMask)
#####

##### conversion code IGBP denomination 
df_file.IGBP_Surface_Type = df_file.IGBP_Surface_Type.apply(decodeIGBP)
#####


dfFiltre = pd.DataFrame(df_file[(df_file.CAD_Score < -20) & ((df_file.ExtinctionQC_532 == 0) | (df_file.ExtinctionQC_532 == 1)) & (df_file.Feature_Optical_Depth_Uncertainty_532 < 99) & ((df_file.FeatureSubtype == 'dust') | (df_file.FeatureSubtype == 'polluted_dust'))])
dfFiltre['Concentration_Aerosols'] = 1000 * (dfFiltre['Column_Optical_Depth_Aerosols_532'] / (dfFiltre['Top_corr'] - dfFiltre['Base_corr']))
################# fin traitement ###########################
###########################################################################





####################### histo filtre #######################
varHisto = ['Latitude', 'Longitude'] + varLissage
dust = df_file[varHisto][df_file.FeatureSubtype==subtypes[0]].values
pdust = df_file[varHisto][df_file.FeatureSubtype==subtypes[1]].values
dustF = df_file[varHisto][df_file.FeatureSubtype==subtypes[0]].values
pdustF = df_file[varHisto][df_file.FeatureSubtype==subtypes[1]].values
fig, ax = plt.subplots(1, len(varHisto), figsize=(23, 12))
fig.text(0.1, 0.5, 'nb valeurs', ha='center', va='center', rotation='vertical')
fig.suptitle(date + 'repartition des valeurs pour les variables ' + ', '.join(varHisto) + 'avant et apres filtre\n(CAD_Score < -20, ExtinctionQC_532 =0 ou 1, Feature_Optical_Depth_Uncertainty < 99)')
for i in range(len(varHisto)):
    ax[i].hist([dust[:,i],dustF[:,i]], color=['green','blue'], ls='dotted', label=['dust', 'dust filtre'], alpha=0.5)
    ax[i].hist([pdust[:,i],pdustF[:,i]], color=['yellow','grey'], ls='dashed', label=['polluted dust', 'polluted dust filtre'], alpha=0.5)
    ax[i].set_xlabel(varHisto[i])
    ax[i].legend(framealpha=0.5)
plt.show()
    
    
    
fig.savefig(ddir_fig + '/'+ date.replace('-', '') + '_histo_filtre.png', dpi=330)

################################################### LISSAGE MEDIANE #####################################################
############utilisation median argMedian ##################################

ww = 9 # 9 15 25 31
# 'Base_corr' 'Top_corr' 'Column_Optical_Depth_Aerosols_532' 'Concentration_Aerosols'
lref = 3
lo = [0,1,2,3] 
lo.remove(lref)

############### lissage prenant en compte les valeurs modifiées en ammont #######################
dftest = dfFiltre[['Latitude','Longitude', 'Base_corr', 'Top_corr', 'Column_Optical_Depth_Aerosols_532', 'Concentration_Aerosols', 'FeatureSubtype']].copy()
params = [p+'_lissage' for p in varLissage]
for c in params:
    dftest[c] = np.nan

for subtype in subtypes:
    arr = dftest[['Base_corr', 'Top_corr', 'Column_Optical_Depth_Aerosols_532', 'Concentration_Aerosols']][dftest.FeatureSubtype == subtype].values
    for i in range(ww/2,arr.shape[0]-(ww/2)):
        med, ix = argMedian(arr[i-((ww-1)/2):i+((ww-1)/2)+1, lref]) # mediane et indice(dans la fenetre de 9 valeurs) de valeur utilisée pour remplacer la valeur cible [i] de la couche de ref
        if arr[i, lref] != med:
            arr[i, lref] = med
            # modif des valeurs des couches suivantes en utilisant les mêmes indices
            arr[i,lo[0]] = arr[i-((ww-1)/2):i+((ww-1)/2)+1,lo[0]][ix] 
            arr[i,lo[1]] = arr[i-((ww-1)/2):i+((ww-1)/2)+1,lo[1]][ix]
            arr[i,lo[2]] = arr[i-((ww-1)/2):i+((ww-1)/2)+1,lo[2]][ix]
    dftest.loc[dftest[dftest.FeatureSubtype == subtype].index, params] = arr[:]




##################################################################################################
##################################################################################################    
##################################################################################################
##################################################################################################
########################### Tests calcul mediane #################################################
varLissage = ['Base_corr', 'Top_corr', 'Column_Optical_Depth_Aerosols_532', 'Concentration_Aerosols']
############# TEST 1 lissage sans prendre en compte les valeurs précédentes modifiées
dftest = dfFiltre[['Latitude','Longitude', 'Base_corr', 'Top_corr', 'Column_Optical_Depth_Aerosols_532', 'Concentration_Aerosols', 'FeatureSubtype']].copy()
params = [p+'_lissage' for p in varLissage]
for c in params:
    dftest[c] = np.nan
for subtype in ['dust', 'polluted_dust']:
    med, ixs = medFilt(dftest.Base_corr[dftest.FeatureSubtype==subtype].values, 9) ## type scipy.signal.medfilt +indices de modif
    for i in range(len(varLissage)):
        dftest.loc[dftest[dftest.FeatureSubtype==subtype].index, varLissage[i]+'_lissage'] = dftest[varLissage[i]][dftest.FeatureSubtype==subtype].values[ixs]

############# TEST 2 prise en compte des valeurs modifiées, intervalle de latitudes (+-0.5°) pris en compte
dftest = dfFiltre[['Latitude','Longitude', 'Base_corr', 'Top_corr', 'Column_Optical_Depth_Aerosols_532', 'Concentration_Aerosols', 'FeatureSubtype']].copy()
params = [p+'_lissage' for p in varLissage]
for c in params:
    dftest[c] = np.nan
for subtype in ['dust', 'polluted_dust']:
    arr = dftest[['Base_corr','Top_corr', 'Column_Optical_Depth_Aerosols_532', 'Concentration_Aerosols']][dftest.FeatureSubtype==subtype].values
    lat = dftest.Latitude[dftest.FeatureSubtype==subtype].values
    dftest.loc[dftest[dftest.FeatureSubtype == subtype].index, params] = lissage1(lat,arr,9,0)
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################    
##################################################################################################


##################################################################################################    
#################################     GRAPHES        #############################################
##################################################################################################

################################### Plots profils ######################################
if lref in [0,1]:
        label = 'Altitude'
elif lref == 2:
    label = 'Epaisseur'
else:
    label = 'Concentration'
fig, axs = plt.subplots(2,1, figsize=(23, 12))
fig.suptitle(date + ' profil de la couche ' + varLissage[lref] + ' pour les ' + ' puis les '.join(subtypes) + ' \navant et apres lissage (fenetre de 9 points)')
for i in range(len(subtypes)):
    axs[i].plot(dftest.Latitude[dftest.FeatureSubtype==subtypes[i]], dftest[varLissage[lref]][dftest.FeatureSubtype==subtypes[i]], color='k', linestyle='-', marker='o', markeredgecolor='black', markerfacecolor='none', markersize=10, label=subtypes[i] + ' init')
    axs[i].plot(dftest.Latitude[dftest.FeatureSubtype==subtypes[i]], dftest[varLissage[lref] + '_lissage'][dftest.FeatureSubtype==subtypes[i]], color='r', linestyle=':', marker='o', markeredgecolor='black',  label=subtypes[i] + ' lissage(9v)')
    axs[i].set_xlabel('Latitude(deg)')
    axs[i].set_ylabel(label)
    axs[i].legend()
plt.show()



fig.savefig(ddir_fig + '/'+ date.replace('-', '') + '_Profil_lissage_' + varLissage[lref] + '.png', dpi=330)

############# ZOOM Profil ########################################
dfsub = dftest[(dftest.Latitude > 8) & (dftest.Latitude < 22)]

fig, ax = plt.subplots(4,1, figsize=(23, 12))
fig.suptitle(date + ' dust et polluted dust confondus:\nlissage (filtre median) ' + varLissage[lref] + ' fenetre (' + str(ww) + ' valeurs)' )
for i in range(4):
    ax[i].plot(dfsub.Latitude,dfsub[varLissage[i]], linestyle='-', color='k')
    ax[i].plot(dfsub.Latitude,dfsub[varLissage[i] + '_lissage'], linestyle='--', color='r')
    ax[i].plot(dfsub.Latitude[dfsub.FeatureSubtype=='dust'],dfsub[varLissage[i]][dfsub.FeatureSubtype=='dust'], linestyle='', marker='o', color='k', markersize=7, markerfacecolor='none', label='dust ' + varLissage[lref] + ' initial')
    ax[i].plot(dfsub.Latitude[dfsub.FeatureSubtype=='dust'],dfsub[varLissage[i]+'_lissage'][dfsub.FeatureSubtype=='dust'], linestyle='', marker='o', markersize=5, color='r', label='dust ' + varLissage[lref] + ' lissage')
    ax[i].plot(dfsub.Latitude[dfsub.FeatureSubtype=='polluted_dust'],dfsub[varLissage[i]][dfsub.FeatureSubtype=='polluted_dust'], linestyle='', marker='x', markersize=7, color='g', markerfacecolor='none', markeredgecolor='k', label='polluted dust ' + varLissage[lref] + ' initial')
    ax[i].plot(dfsub.Latitude[dfsub.FeatureSubtype=='polluted_dust'],dfsub[varLissage[i]+'_lissage'][dfsub.FeatureSubtype=='polluted_dust'], linestyle='', marker='x', markersize=5, color='b', markeredgecolor='r', label='polluted dust ' + varLissage[lref] + ' lissage')
    if varLissage[i] in ['Base_corr', 'Top_corr']:
        ax[i].set_ylabel('Altitude(m)')
    elif varLissage[i] in ['Column_Optical_Depth_Aerosols_532']:
        ax[i].set_ylabel('Epaisseur')
    else:
       ax[i].set_ylabel('Concentration') 
    ax[i].legend(framealpha=0.5)
plt.show()




fig.savefig(ddir_fig + '/'+ date.replace('-', '') + '_zoom_Profil_lissage_' + varLissage[lref] + '.png', dpi=330)

##########################################################################################
########## histogrammes ##################################################################
Stypes = [['dust','polluted_dust'],['dust'],['polluted_dust']]
fig, axs = plt.subplots(3, 4, figsize=(23, 12))
fig.suptitle(date + ' repartition des valeurs de ' + '/'.join(['+'.join(St) for St in Stypes]) + ' pour chaque variable avant et apres lissage(lissage ref ' + varLissage[lref] + ')')
for i in range(3):
    axs[i,0].hist(dftest[varLissage[0]][dftest.FeatureSubtype.isin(Stypes[i])].values, bins=range(10), color='black', label=varLissage[0], alpha=0.5)
    axs[i,0].hist(dftest[varLissage[0]+'_lissage'][dftest.FeatureSubtype.isin(Stypes[i])].values, bins=range(10), color='blue', label=varLissage[0] + ' lissage', alpha=0.5)
    axs[i,0].set_xlabel('Altitude(km)')
    axs[i,0].set_ylabel('Frequence ' + '+'.join(Stypes[i]))
    axs[i,0].legend(framealpha=0.5)
    #axs[i,1].set_title(date + ' repartition des valeurs de ' + '/'.join(['+'.join(St) for St in Stypes]) + ' pour chaque variable avant et apres lissage(lissage ref ' + varLissage[lref] + ')')
    axs[i,1].hist(dftest[varLissage[1]][dftest.FeatureSubtype.isin(Stypes[i])].values, bins=np.arange(10), color='black', label=varLissage[1], alpha=0.5)
    axs[i,1].hist(dftest[varLissage[1]+'_lissage'][dftest.FeatureSubtype.isin(Stypes[i])].values, bins=np.arange(10), color='blue', label=varLissage[1] + ' lissage', alpha=0.5)
    axs[i,1].set_xlabel('Altitude(km)')
    axs[i,1].legend(framealpha=0.5)
    axs[i,2].hist(dftest[varLissage[2]][dftest.FeatureSubtype.isin(Stypes[i])].values, bins=np.arange(0,2,0.1), color='black', label='Col AOD', alpha=0.5)
    axs[i,2].hist(dftest[varLissage[2]+'_lissage'][dftest.FeatureSubtype.isin(Stypes[i])].values, bins=np.arange(0,2,0.1), color='blue', label='Col AOD lissage', alpha=0.5)
    axs[i,2].set_xlabel('Epaisseur')
    axs[i,2].legend()
    axs[i,3].hist(dftest[varLissage[3]][dftest.FeatureSubtype.isin(Stypes[i])].values, bins=range(0, 1400,100), color='black', label=varLissage[3], alpha=0.5)
    axs[i,3].hist(dftest[varLissage[3]+'_lissage'][dftest.FeatureSubtype.isin(Stypes[i])].values, bins=np.arange(0,1400,100), color='blue', label=varLissage[3] + ' lissage', alpha=0.5)
    axs[i,3].set_xlabel('Concentration')
    axs[i,3].legend(framealpha=0.5)
plt.show()




fig.savefig(ddir_fig + '/'+ date.replace('-', '') + '_Histo_lissage_' + varLissage[lref] + '.png', dpi=330)
