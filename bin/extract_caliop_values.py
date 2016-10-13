# -*- coding: utf-8 -*-

from pyhdf.SD import SD, SDC
from netCDF4 import Dataset, date2num, date2index
import rasterio
from geopandas import GeoDataFrame
from shapely.geometry import Point, shape
import fiona
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
path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
#path = os.getcwd()
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
layer = 'Concentration_Aerosols' ## 'Base_corr', 'Top_corr', 'Column_Optical_Depth_Aerosols_532', 'Concentration_Aerosols'
liss = 'lissage' + layer + str(w_lissage) + 'v'
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
layerInit = 'Layer_Base_Altitude'
#layers_ref = ['Base_corr', 'Top_corr', 'Column_Optical_Depth_Aerosols_532']



# liste des variables/parametres extraits de chaque fichier lidar
varlist = ["IGBP_Surface_Type", "Day_Night_Flag", "DEM_Surface_Elevation", "Column_Optical_Depth_Aerosols_532", "Feature_Optical_Depth_532","Feature_Optical_Depth_Uncertainty_532", "ExtinctionQC_532", "CAD_Score", "Feature_Classification_Flags", "Number_Layers_Found","Layer_Base_Extended", "Relative_Humidity", "Layer_Top_Altitude", "Layer_Base_Altitude"]

# Variables/parametres sur lesquelles lissees
param_lissage = ['Base_corr', 'Top_corr', 'Column_Optical_Depth_Aerosols_532', 'Concentration_Aerosols']

# liste des variables/parametres interpolees pour chaque sous-type
params_export = param_lissage + ['Feature_Optical_Depth_Uncertainty_532', 'Feature_Optical_Depth_532', 'Relative_Humidity']

lref = param_lissage.index(layer)


cpu = 3  # joblib.cpu_count() - 1 # nombre de processeurs pour la parallelisation

#########################################################################################################################

#######################################################################
########## traitement dates ###########################################
# creation de pas de temps de n jours(ptemps) avec comme origine le 1900-01-01

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


gtif = rasterio.open(path + '/src/mask/maskAfrica_025deg.tif')
rast = np.squeeze(gtif.read())
rast[rast == gtif.nodata] = np.nan
shp = fiona.open(path + '/src/mask/maskAfrica.shp')[0]

################################################################

t = time.time()
# boucle pour chaque periode de n jours
for k in sorted(dt_nday.keys())[idt:]:
    t1 = time.time()
    print "\n\n\n###########  periode   du", k.date(), " au ", (k + timedelta(days=ptemps-1)).date()
    print "#########################################################"
    df_nday = pd.DataFrame()  # creation d une dataframe pour n jours
    #####  boucle pour chaque date de la periode de n jours
    for day in dt_nday[k][:]:
        #df_day = pd.DataFrame() # initialisation dataframe pour une orbite/jour
        ##### boucle pour chaque fichier de la date
        for idf, f in enumerate(files[day][:]):
            hdf = SD(f, SDC.READ)
            df_file = pd.DataFrame()  # dataframe temporaire reinitialisee pour chaque fichier
            df_file["Latitude"] = hdf.select('Latitude')[:, 1]
            df_file["Longitude"] = hdf.select('Longitude')[:, 1]
            df_file["Date"] = day
            df_file["idFile"] = f[-14:-4]
            ##### Extraction de l'indice de la 1ere couche valide en s appuyant sur la 1ere variable definie dans layers_ref
            base = hdf.select(layerInit)
            base_mat = base[:]
            list_mat_in = np.vsplit(base_mat, base_mat.shape[0])  # decoupage de la matrice 1D en une liste de n sous-matrices de dim (1,8)
            list_mat_out = map(indiceCouche, list_mat_in)  # appel fonction indiceCouche pour recuperer l'indice de la 1ere couche valide pour chaque coord lat/lon      
            df_file[layerInit] = [m[0] for m in list_mat_out]
            indices = [m[1] for m in list_mat_out]  # liste des indices qui seront ensuite utilisés comme 'masque' pour les autres couches
            base.endaccess()
            #####
            
            ##### boucle d'extraction des valeurs de chaque variable correspondant aux indices definis precedemment
            for var in list(set(varlist) - set([layerInit])):  # 1ere variable de layers_ref exclue
                try:
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
                except KeyError:
                    df_file[var] = np.nan
                var_dict.endaccess()
            hdf.end()
            df_file = df_file[df_file.Layer_Base_Altitude != -9999]  # suppression des points sans valeur
            
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

            ##### filtre qualite : CAD-Score < -20, ExtinctionQC_532 = 0 ou = 1, Feature_Optical_Depth_Uncertainty < 99, suppression des subtypefeature non aerosols
            dfFiltre = df_file[(df_file.CAD_Score < -20) & ((df_file.ExtinctionQC_532 == 0) | (df_file.ExtinctionQC_532 == 1)) & (df_file.Feature_Optical_Depth_Uncertainty_532 < 99) & ((df_file.FeatureSubtype == 'dust') | (df_file.FeatureSubtype == 'polluted_dust'))].copy()
            #####

            ##### calcul concentration en aerosol
            cst = 1
            try:
                dfFiltre['Concentration_Aerosols'] = 1000 * (dfFiltre['Column_Optical_Depth_Aerosols_532'] / (dfFiltre['Top_corr'] - dfFiltre['Base_corr'])) * cst
            except AttributeError:
                pass
            #####

            params = [p+'_lissage' for p in param_lissage]
            for c in params:
                dfFiltre[c] = np.nan
            try:
                for subtype in subtypes:
                    ##### lissage (mediane) n fois en fct du nombre de variables choisies(layers_ref)
                    tmp = lissage(dfFiltre[dfFiltre.FeatureSubtype == subtype][param_lissage].values, w_lissage, lref)
                    dfFiltre.loc[dfFiltre[dfFiltre.FeatureSubtype == subtype].index, params] = tmp
            except AttributeError:
                pass
            df_nday = df_nday.append(dfFiltre[(dfFiltre.Latitude >= yo.min()) & (dfFiltre.Latitude <= yo.max()) & (dfFiltre.Longitude >= xo.min()) & (dfFiltre.Longitude <= xo.max())],ignore_index=True) # extraction de la zone d'etude
    df_nday['idPeriod'] = k

    ##### export csv mask Afrique
#    geoms = [Point(xy) for xy in zip(df_nday.Longitude,df_nday.Latitude)]
#    mask = df_nday.index[[geom.within(shape(shp['geometry'])) for geom in geoms]]
#    df_africa = df_nday.ix[mask]
    ##### export format csv
    df_nday.to_csv(ddir_out+'/'+k.strftime("%Y_%m_%d")+'_'+str(ptemps)+'d_' + liss + '.csv', index=False)  
    #####
    print('%s sec' % str(time.time()-t1))
    ##### interpolation
    cols = ['FeatureSubtype', 'Longitude', 'Latitude'] + params + ['Feature_Optical_Depth_Uncertainty_532', 'Feature_Optical_Depth_532', 'Relative_Humidity'] #liste des parametres exportées avec le suffixe correspondant au dernier lissage
    df_nday_out = df_nday[cols].reset_index(drop=True) # extraction de la subdataframe
    for c in df_nday_out.columns:
        if 'lissage' in c:
            df_nday_out.rename(columns={c: c[:-9]}, inplace=True)
    output = extractData(methode_ponderation, params_export, df_nday_out, fichiers_ext[:], k, w_interp, cpu, xo, yo, reso_spatiale)
    #####
    
    ##### export format csv
    #exportTXT(params_export, fichiers_ext, output, k, 'dust')
    #####

    ##############################
    # create netcdf4##############
    u_time = 'hours since 1900-01-01 00:00:00.0'
    dates = [date2num(k, u_time)]  # date2num(sorted(dt_nday.keys())[idt:],u_time)
    
    ncnew = Dataset(ddir_out+'/'+str(k.date()).replace('-','_')+'_lidar_'+methode_ponderation+'_'+fenetre+'_'+str(ptemps)+'d_' + liss + '.nc', 'w')
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
    lidar_stat = ['', '_min', '_max', '_std']
    lidar_keys = [p+'_point2grid' for p in params_export] + ['nb_lidar_points'] + [p + sts for sts in lidar_stat for p in params_export]
    ext_stat = ['_pct_px', '_mean', '_min', '_max', '_std']
    ext_keys = [v + sts for sts in ext_stat for v in var_ext]
    try:
        idx = ext_keys.index('mean_pDUST_mean')
        ext_keys[idx] = 'pDUST_mean'
    except ValueError:
        pass
        
    for subtype in subtypes:
        for i in range(len(lidar_keys)):
            varnew = ncnew.createVariable(subtype + '_' + lidar_keys[i], 'f4', ('time', 'latitude', 'longitude'), fill_value=fillvalue)
            varnew.standard_name = subtype + '_' + lidar_keys[i]
            varnew[idate, ...] = output[subtype][:, i].reshape(yo.shape[0], -1) * rast
    for key in var_ext:
        ext_stat = ['_pct_px', '_mean', '_min', '_max', '_std']
        for j in range(5):
            if key == 'mean_pDUST':
                varnew = ncnew.createVariable(key[5:]+ext_stat[j], 'f4', ('time', 'latitude', 'longitude'), fill_value=fillvalue)
                varnew.standard_name = key[5:]+ext_stat[j]
            else:
                varnew = ncnew.createVariable(key+ext_stat[j], 'f4', ('time', 'latitude', 'longitude'), fill_value=fillvalue)
                varnew.standard_name = key+ext_stat[j]
            varnew[idate,...] = output[key][:,j].reshape(yo.shape[0],-1) * rast

    ncnew.close()
    print('%s sec' % str(time.time() - t1))
print('%s sec' % str(time.time() - t))

#try:
#    year = sorted(dt_nday.keys())[idt + 1].year
#except IndexError:
#    year = sorted(dt_nday.keys())[idt].year
#popen = Popen([path+'/src/concat_lidar.sh', ddir_out, str(year), methode_ponderation, fenetre, str(ptemps), str(w_lissage)])
