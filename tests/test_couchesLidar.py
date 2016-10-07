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


# params pandas 
# pd.options.display.max_rows = 999
# pd.set_option('expand_frame_repr', False)


# import des fonctions complémentaires
path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
#path = os.getcwd()
sys.path.append(path+"/src")
from rolling_window import *
from LidarUtil import *

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
df_file[layerInit] = [m[0] for m in list_mat_out]
Lbase[Lbase == -9999] = np.nan
 = Lbase
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


            params = [p+'_lissage' for p in param_lissage]
            for c in params:
                df_file[c] = np.nan
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
                #####
                #lp = list(set(df_file.columns) - set(['Layer_Base_Altitude', 'Layer_Top_Altitude']))
                ##### filtre qualite : CAD-Score < -20, ExtinctionQC_532 = 0 ou = 1, Feature_Optical_Depth_Uncertainty < 99, suppression des subtypefeature non aerosols
                df_nday = df_nday.append(df_file[(df_file.CAD_Score < -20) & ((df_file.ExtinctionQC_532 == 0) | (df_file.ExtinctionQC_532 == 1)) & (df_file.Feature_Optical_Depth_Uncertainty_532 < 99) & ((df_file.FeatureSubtype == 'dust') | (df_file.FeatureSubtype == 'polluted_dust'))], ignore_index=True)
                #####
            except AttributeError:
                pass
    df_nday['idPeriod'] = k
    ##### calcul concentration en aerosol
    try:
        df_nday['Concentration_Aerosols'] = 1000 * (df_nday['Column_Optical_Depth_Aerosols_532_lissage'] / (df_nday['Top_corr_lissage'] - df_nday['Base_corr_lissage'])) * 1
    except AttributeError:
        pass
    #####
    ##### export csv mask Afrique
#    geoms = [Point(xy) for xy in zip(df_nday.Longitude,df_nday.Latitude)]
#    mask = df_nday.index[[geom.within(shape(shp['geometry'])) for geom in geoms]]
#    df_africa = df_nday.ix[mask]
    ##### export format csv
    df_nday.to_csv(ddir_out+'/'+k.strftime("%Y_%m_%d")+'_'+str(ptemps)+'d_' + liss + '_test.csv', index=False)  
    #####
    print('%s sec' % str(time.time()-t1))
    ##### interpolation
    cols = ['FeatureSubtype', 'Longitude', 'Latitude', 'Concentration_Aerosols'] + params #liste des parametres exportées avec le suffixe correspondant au dernier lissage
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
    
    ncnew = Dataset(ddir_out+'/'+str(k.date()).replace('-','_')+'_lidar_'+methode_ponderation+'_'+fenetre+'_'+str(ptemps)+'d_' + liss + '_test.nc', 'w')
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
#popen = Popen([path+'/src/concat_lidar.sh', ddir_out, str(year), methode_ponderation, fenetre, str(ptemps), w_lissage])
