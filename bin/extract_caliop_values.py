# -*- coding: utf-8 -*-

from pyhdf.SD import SD, SDC
from netCDF4 import Dataset, date2num, date2index
import time
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, timedelta
from bisect import bisect_left
import os
import sys
from subprocess import Popen
from joblib import cpu_count

# import des fonctions complémentaires
#path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
path = os.getcwd()
sys.path.append(path+"/src")
from rolling_window import *
from LidarUtil import *


#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
ddir_in = path+"/zone_etude"
ddir_out = path+"/out"
if os.path.isdir(ddir_in):
	os.chdir(ddir_in)
else:
	os.mkdir(ddir_in)
	print 'Aucun dossier lidar source, '+ddir_in+' cree'
if not os.path.isdir(ddir_out):
	os.mkdir(ddir_out)
	print 'dossier de sortie '+ddir_out+' cree'


#########################################################################################################################
#########################################################################################################################
#########################################################################################################################


################ Parametres de fonctions modifiables ####################################################################


# zone d'etude
x_min, x_max = -25.0, 57.01
y_min, y_max = -1.25, 51.01
reso_spatiale = 0.25
xo = np.arange(x_min, x_max, reso_spatiale)  #longitudes du .nc
yo = np.arange(y_min, y_max, reso_spatiale)[::-1]  #latitudes du .nc


w_lissage = 9 # impair fenetre pour fonction de lissage
w_interp = 9 # 3 6 9 12 15 ... fenetre glissante pour l'interpolation
fenetre = str(w_interp)+'px'


ptemps = 16  # pas de temps
methode_ponderation = 'carredistance' #'distance', 'carredistance'
subtypes = ['dust', 'polluted_dust'] # sous-types d'aerosols extraits des donnees lidar


###### liste des donnees externes integrees au fichier netcdf en sortie
path_ext = path+'/donnees_annexes/'
# aod aerus 0.05 deg
fnc1 = path_ext+'seviri_r005_16d.nc'
var1 = 'AOD_VIS06'
# aod MYD04 0.09 deg
fnc2 = path_ext+'MYD04_r009_16d.nc'
var2 = 'Deep_Blue_Aerosol_Optical_Depth_550_Land'
# AI omaeruv 0.25 deg
fnc3 = path_ext+'omaeruv_r025_16d.nc'
var3 = 'UVAerosolIndex'
# pDUST chimere02 0.18 deg
fnc4 = path_ext+'chimere02_r018_16d_subset.nc'
var4 = 'mean_pDUST'

#fnc... = path_ext+'nom_du_fichier'
#var... = 'nom_de_la_variable'
# Ajouter fnc... et var... dans la liste fichiers_ext
fichiers_ext = [[fnc1, var1], [fnc2, var2], [fnc3, var3], [fnc4, var4]]
var_ext = [v[1] for v in fichiers_ext]



##### Parametres lies aux fichiers d'entree lidar

# !!!!!! la 1ere variable de la liste layers_ref sert de couche de reference pour l'extraction de la premiere couche valide du profil lidar et egalement utilisee pour l'etape de lissage
# Les variables ajoutées a la suite seront prises en compte lors de l'etape de lissage
layers_ref = ['Layer_Base_Altitude', 'Layer_Top_Altitude']




# liste des variables/parametres extraits de chaque fichier lidar
varlist = ["IGBP_Surface_Type", "Day_Night_Flag", "DEM_Surface_Elevation", "Column_Optical_Depth_Aerosols_532", "Feature_Optical_Depth_532","Feature_Optical_Depth_Uncertainty_532", "ExtinctionQC_532", "CAD_Score", "Feature_Classification_Flags", "Number_Layers_Found", "Layer_Top_Altitude", "Layer_Base_Altitude"]

# Variables/parametres sur lesquelles lissees
param_lissage = ['Column_Optical_Depth_Aerosols_532', 'Base_corr', 'Top_corr', 'Layer_Base_Altitude', 'Layer_Top_Altitude', 'Feature_Optical_Depth_532', 'Feature_Optical_Depth_Uncertainty_532', 'ExtinctionQC_532']

# liste des variables/parametres interpolees pour chaque sous-type
params_export = ["Column_Optical_Depth_Aerosols_532", "Feature_Optical_Depth_532", "Feature_Optical_Depth_Uncertainty_532", "Top_corr", "Base_corr", "Concentration_Aerosols"]




cpu = 3  # joblib.cpu_count() - 1 # nombre de processeurs pour la parallelisation

#########################################################################################################################

#######################################################################
########## traitement dates ###########################################
# creation de pas de temps de n jours(ptemps) en commencant le 1900-01-01

files_in = sorted(glob("*.hdf"))
if not files_in:
    print 'Absence de fichiers lidar'
    sys.exit()
try:
    debut = datetime.strptime(files_in[0][31:41], "%Y-%m-%d")
    fin = datetime.strptime(files_in[-1][31:41], "%Y-%m-%d")
except ValueError:
    print 'Impossible de lire la date du fichier'
    sys.exit()
series = pd.date_range('1900-01-01',fin, freq="d").tolist() # serie de dates de 1900-01-01 a date de fin des fichiers 
dt_nday = {series[i]:series[i:i+ptemps] for i in range(0,len(series),ptemps)} # dict regroupant les dates par serie de n jours(ptemps) , la clef du dict correspond a la 1ere date chaque periode de n jours
idt = bisect_left(sorted(dt_nday.keys()),debut) - 1 # a partir de la 1ere date de la liste de fichiers determination de la clef inferieure la plus proche dans le dict 
idt_series = series.index(sorted(dt_nday.keys())[idt]) # index de la date de la 1ere periode de njours dans la serie de dates depuis 1900-01-01
files = {n:sorted(glob("*"+n.strftime('%Y-%m-%d')+"*.hdf")) for n in series[idt_series:]} # extraction des fichiers dans un dict, chaque entree corespond a une date 

################################################################

t1 = time.time()
# boucle pour chaque periode de n jours
for k in sorted(dt_nday.keys())[idt:]:
    t2 = time.time()
    print "\n\n\n###########  periode   du", k.date(), " au ", (k + timedelta(days=ptemps-1)).date()
    print "#########################################################"
    df_nday = pd.DataFrame()  # creation d une dataframe pour n jours
    #####  boucle pour chaque date de la periode de n jours
    for day in dt_nday[k][:]:
        df_day = pd.DataFrame() # initialisation dataframe pour une orbite/jour
        ##### boucle pour chaque fichier de la date
        for f in files[day][:]:
            hdf = SD(f, SDC.READ)
            df_file = pd.DataFrame()  # dataframe temporaire reinitialisee pour chaque fichier
            df_file["Latitude"] = hdf.select('Latitude')[:, 1]
            df_file["Longitude"] = hdf.select('Longitude')[:, 1]
            df_file["Date"] = day
            
            ##### Extraction de l'indice de la 1ere couche valide en s appuyant sur la 1ere variable definie dans layers_ref
            base = hdf.select(layers_ref[0])
            base_mat = base[:]
            list_mat_in = np.vsplit(base_mat, base_mat.shape[0])  # decoupage de la matrice 1D en une liste de n sous-matrices de dim (1,8)
            list_mat_out = map(indiceCouche, list_mat_in)  # appel fonction indiceCouche pour recuperer l'indice de la 1ere couche valide pour chaque coord lat/lon      
            df_file[layers_ref[0]] = [m[0] for m in list_mat_out]
            indices = [m[1] for m in list_mat_out]  # liste des indices qui seront ensuite utilisés comme 'masque' pour les autres couches
            base.endaccess()
            #####
            
            ##### boucle d'extraction des valeurs de chaque variable correspondant aux indices definis precedemment
            for var in list(set(varlist) - set([layers_ref[0]])):  # 1ere variable de layers_ref exclue
                var_dict = hdf.select(var)
                mat_in = var_dict[:]
                attribs = var_dict.attributes()
                if hdf.datasets()[var][1][1] == 1:
                    df_file[var] = mat_in[:].flatten()
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
                var_dict.endaccess()
            hdf.end()
            df_file = df_file[df_file.Layer_Base_Altitude != -9999]  # suppression des points sans valeur

            ##### Conversion int16 en sous-categories
            df_file['FeatureSubtype'] = df_file.Feature_Classification_Flags.apply(decodeFeatureMask)
            #####
            
            ##### conversion code IGBP denomination 
            df_file.IGBP_Surface_Type = df_file.IGBP_Surface_Type.apply(decodeIGBP)
            #####

            ##### calcul de Layer_Top_Altitude et Layer_Base_Altitude corrigées a partir du DEM
            df_file['Top_corr'] = df_file.Layer_Top_Altitude - df_file.DEM_Surface_Elevation
            df_file['Base_corr'] = df_file.Layer_Base_Altitude - df_file.DEM_Surface_Elevation
            #####

            ##### filtre qualite : CAD-Score < -20, ExtinctionQC_532 = 0 ou = 1, Feature_Optical_Depth_Uncertainty < 99, suppression des subtypefeature non aerosols
            df_day = df_day.append(df_file[(df_file.CAD_Score < -20) & ((df_file.ExtinctionQC_532 == 0) | (df_file.ExtinctionQC_532 == 1)) & (df_file.Feature_Optical_Depth_Uncertainty_532 < 99) & (df_file.FeatureSubtype != 'no_aerosol')], ignore_index=True)
            #####

            ##########################################################################################################################################

        ##### calcul_Nbpoints pour definir le nombre de points non nuls dans +-0.5 deg de latitude autour de chaque point ####
        df_day['nb_values_window_1deg'] = np.nan
        try:
            for subtype in df_day.FeatureSubtype.unique().tolist():
                matrice_latitude = df_day.Latitude[df_day.FeatureSubtype == subtype].values
                df_day['nb_values_window_1deg'][df_day.FeatureSubtype == subtype ].values[:] = [calcul_Nbpoints(matrice_latitude, vx) for vx in list(matrice_latitude)]
                ##### lissage (mediane) n fois en fct du nombre de variables choisies(layers_ref)
                for lref in layers_ref:
                    df_day[df_day.FeatureSubtype == subtype][param_lissage].values[:] = lissage(df_day[df_day.FeatureSubtype == subtype][param_lissage], w_lissage, lref).values
            #####
    
            ##### ajout de la dataframe par orbite, correspondant a l'ensemble des variables lidar filtrées, dans la dataframe de n jours
            lp = list(set(df_day.columns) - set(['Feature_Classification_Flags', 'Layer_Top_Altitude', 'Layer_Base_Altitude']))
            df_nday = df_nday.append(df_day[lp][(df_day.Latitude >= yo.min()) & (df_day.Latitude <= yo.max()) & (df_day.Longitude >= xo.min()) & (df_day.Longitude <= xo.max())],ignore_index=True) # extraction de la zone d'etude
        except AttributeError:
            pass
        ##############################################################################################################################################
                                            ############# fin des pre-traitements ######################
        ##############################################################################################################################################

    ##### calcul concentration en aerosol
    try:
        df_nday['Concentration_Aerosols'] = 1000 * (df_nday.Column_Optical_Depth_Aerosols_532 / (df_nday.Top_corr - df_nday.Base_corr)) * 1
    except AttributeError:
        pass
    #####
    
    ##### export format csv
    df_nday.to_csv(ddir_out+'/'+k.strftime("%Y_%m_%d")+'_'+methode_ponderation+'_'+fenetre+'.csv', index=False)  
    #####
    print('%s sec' % str(time.time()-t2))
    ##### interpolation
    output = extractData(methode_ponderation, params_export, df_nday[df_nday.FeatureSubtype.isin(subtypes)].reset_index(drop=True), fichiers_ext[:], k, w_interp, cpu, xo, yo, reso_spatiale)
    #####
    
    ##### export format csv
    #exportTXT(params_export, fichiers_ext, output, k, 'dust')
    #####

    ##############################
    # create netcdf4##############
    u_time = 'hours since 1900-01-01 00:00:00.0'
    dates = [date2num(k, u_time)]  # date2num(sorted(dt_nday.keys())[idt:],u_time)
    
    ncnew = Dataset(ddir_out+'/'+str(k.date()).replace('-','_')+'_lidar_'+methode_ponderation+'_'+fenetre+'.nc', 'w')
    # dimensions##################
    ncnew.createDimension('time', None)
    ncnew.createDimension('latitude', len(yo))
    ncnew.createDimension('longitude', len(xo))
    # variables####################
    tp = ncnew.createVariable('time','f8',('time',))
    lats = ncnew.createVariable('latitude','f4',('latitude',))
    lons = ncnew.createVariable('longitude','f4',('longitude',))
    # attributs###################
    ncnew.Convention ='CF-1.5'
    ncnew.description = ''
    ncnew.history = ''
    ncnew.source = ''
    lats.units = 'degrees_north'
    lats.standard_name = 'latitude'
    lons.units = 'degrees_east'
    lons.standard_name = 'longitude'
    tp.units = u_time
    tp.standard_name = 'time'
    tp.calendar = 'gregorian'
    # write values#################
    tp[:] = dates[:]
    lats[:] = yo[:]
    lons[:] = xo[:]
    fillvalue = np.nan

    idate = 0  # date2index(k,tp)  # indice de la date de la donnee dans le netcdf
    lidar_keys = sorted(list(set(output.keys()) - set(var_ext)))
    for key in lidar_keys:
        if 'nb_lidar_points' in key:
            varnew = ncnew.createVariable(key, 'f4', ('time', 'latitude', 'longitude'), fill_value=fillvalue)
            varnew.standard_name = key
            varnew[idate,...] = output[key].reshape(yo.shape[0], -1)
        else:
            calc = ['', '_min', '_max', '_std']
            for j in range(4):
                varnew = ncnew.createVariable(key+calc[j], 'f4', ('time', 'latitude', 'longitude'), fill_value=fillvalue)
                varnew.standard_name = key+calc[j]
                varnew[idate, ...] = output[key][:, j].reshape(yo.shape[0], -1)
    for key in var_ext:
        calc = ['_pct_px', '_mean', '_min', '_max', '_std']
        for j in range(5):
            if key == 'mean_pDUST':
                varnew = ncnew.createVariable(key[5:]+calc[j], 'f4', ('time', 'latitude', 'longitude'), fill_value=fillvalue)
                varnew.standard_name = key[5:]+calc[j]
            else:
                varnew = ncnew.createVariable(key+calc[j], 'f4', ('time', 'latitude', 'longitude'), fill_value=fillvalue)
                varnew.standard_name = key+calc[j]
            varnew[idate,...] = output[key][:,j].reshape(yo.shape[0],-1)

    ncnew.close()
print('%s sec' % str(time.time() - t1))

popen = Popen([path+'/bin/concat_lidar.sh', ddir_out, methode_ponderation, fenetre])
