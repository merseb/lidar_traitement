# -*- coding: utf-8 -*-

from netCDF4 import Dataset, num2date, date2num, date2index
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import pandas as pd
import multiprocessing
import time
from datetime import datetime

def splitlist(a, n):
    """
    divise liste a en n sous-listes +- egales
    """
    k, m = len(a) / n, len(a) % n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))


#def extractIndices(matrice, longitude, latitude, coords, window, reso=0.25):
#    """
#    
#    """
#    x_px = coords[0]
#    y_px = coords[1]
#    # definition de la fenetre
#    x_min = x_px - reso*(window/2)
#    x_max = x_px + reso*(window/2)
#    y_min = y_px - reso*(window/2)
#    y_max = y_px + reso*(window/2)
#    idx_prod = list(np.where((latitude >= y_min) & (latitude < y_max) & (longitude >= x_min) & (longitude < x_max))[0])
#    # return idx_prod
#    mat_valeurs = matrice[idx_prod]
#    nbpx = mat_valeurs.shape[0]
#    nbpx_nonull = np.count_nonzero(~np.isnan(mat_valeurs))
#    pourcent_px = 100*(nbpx_nonull/nbpx)
#    std = np.nanstd(mat_valeurs)
#    mini = np.nanmin(mat_valeurs)
#    maxi = np.nanmax(mat_valeurs)
#    moy = np.nanmean(mat_valeurs)
#    return pourcent_px, moy, mini, maxi, std


def extractData(matrice, longitude, latitude, coords, window, reso=0.25):
    """
    
    """
    pourcent_px = []
    std = []
    mini = []
    maxi = []
    moy = []
    for coord in coords:
        x_px = coord[0]
        y_px = coord[1]
        # definition de la fenetre
        x_min = x_px - reso*(window/2)
        x_max = x_px + reso*(window/2)
        y_min = y_px - reso*(window/2)
        y_max = y_px + reso*(window/2)
        idx_prod = list(np.where((latitude >= y_min) & (latitude < y_max) & (longitude >= x_min) & (longitude < x_max))[0])
        mat_valeurs = matrice[idx_prod]
        nbpx = mat_valeurs.shape[0]
        nbpx_nonull = np.count_nonzero(~np.isnan(mat_valeurs))
        pourcent_px.append(100*(nbpx_nonull/nbpx))
        std.append(np.nanstd(mat_valeurs))
        mini.append(np.nanmin(mat_valeurs))
        maxi.append(np.nanmax(mat_valeurs))
        moy.append(np.nanmean(mat_valeurs))
    return pourcent_px, std, mini, maxi, moy


def extractData2(path_mat, variable, date, window, cpu, reso=0.25):
    """
    
    filtre extraction stat 
    
    PARAMETRES:
    
    **path_mat**(*string*): chemin pointant sur fichier nc
    **variable**(*string*): nom de la variable traitee
    **date** (*datetetime object*): date
    **window** (*int*): largeur de la fenetre glissante
    **cpu** (*int*): nombre de processeurs pour la parallelisation
    **reso** (*float*): par defaut 0.25 deg, resolution spatiale de la donnee en sortie
    
    """
    t1 = time.time()
    xo = np.arange(-25.,57.01, reso)  #longitudes du .nc
    yo = np.arange(-1.3,51.01, reso)[::-1]  #latitudes du .nc
    xx,yy = np.meshgrid(xo,yo)
    xy = zip(xx.flatten(),yy.flatten())    
    
    nc = Dataset(path_mat, 'r')
    ncdates = nc.variables['time']
    id_date = date2index(date, ncdates, calendar=ncdates.calendar, select='after')
    mat = nc.variables[variable][id_date, ...]
    lg = nc.variables['longitude'][:]
    lt = nc.variables['latitude'][:]
    lons, lats = np.meshgrid(lg, lt)
    list_jobs = [job for job in splitlist(xy, cpu)]
    list_values = Parallel(n_jobs=cpu)(delayed(extractData)(mat.flatten(), lons.flatten(), lats.flatten(), lcoords, window) for lcoords in list_jobs)
    t2 = time.time() - t1
    print t2
    return list_values  # [extractIndices(mat, lg, lt, c, ww) for c in coordonnees]

#def fct_helper(matrice, lgs, lts, lcoords, window):
#    return [extractIndices(matrice, lgs, lts, cds, window, reso=0.25) for cds in lcoords]
#
##def fct_helper(args):
##    return extractData(*args)
#
#
#def parallelJobs(matrice, longitude, latitude, list_jobs, window, reso=0.25):
#    p = multiprocessing.Pool(3)
#    job_args = [(matrice, longitude, latitude, j, window, reso) for j in list_jobs]
#    return p.map(fct_helper, job_args)


if __name__ == "__main__":

#    fcsv = "/home/mers/Bureau/teledm/donnees/lidar/calipso/caliop/out/flag_test_filter_2014_01_27.csv"
#    csv = pd.read_csv(fcsv, header=0)
#    df_alldust = csv[(csv.FeatureType == 3) & (csv.FeatureTypeQA > 1)]
#    df_dust = csv[(csv.FeatureType == 3) & (csv.FeatureTypeQA > 1) & (csv.FeatureSubtype == 2)]
#    df_polluteddust = csv[(csv.FeatureType == 3) & (csv.FeatureTypeQA > 1) & (csv.FeatureSubtype == 5)]
#    lon = df_dust.Longitude.values
#    lat = df_dust.Latitude.values
#    datas = df_dust.Layer_Base_Altitude.values
#    xy = zip(lon,lat)
#    
#    fnc = '/home/mers/Bureau/teledm/donnees/satellite/msg/seviri_aerus/res005/seviri_r005_16d.nc'
#    nc = Dataset(fnc,'r')
#    lati = nc.variables['latitude'][:]
#    longi = nc.variables['longitude'][:]
#    aod = nc.variables['AOD_VIS06'][0,...].flatten()
#    lons,lats = np.meshgrid(longi,lati)
#
#    w_interp = 9
#    list_jobs = [job for job in splitlist(xy[:],3)]
##    t1 = time.time()
##    list_values1 = Parallel(n_jobs=3)(delayed(extractData)(aod.flatten(),lons.flatten(),lats.flatten(),l_coord,w_interp) for l_coord in list_jobs)
##    t2 = time.time() - t1
#    t1 = time.time()
#    list_values2 = parallelJobs(aod.flatten(),lons.flatten(),lats.flatten(),list_jobs,w_interp,0.25)
#    t2 = time.time() - t1
##    t1 = time.time()
##    list_values3 = [extractIndices(aod.flatten(),lons.flatten(),lats.flatten(),coords,w_interp) for coords in xy]
##    t2 = time.time() - t1
#    t1 = time.time()
#    list_values4 = Parallel(n_jobs=3)(delayed(extractData2)(aod.flatten(),lons.flatten(),lats.flatten(),l_coord,w_interp) for l_coord in list_jobs)
#    t2 = time.time() - t1
    w_interp = 9
    cpu = 3
    v_path = '/home/mers/Bureau/teledm/donnees/satellite/msg/seviri_aerus/res005/seviri_r005_16d.nc'
    d = datetime(2014, 1, 1)
    lval = extractData2(v_path, 'AOD_VIS06', d, w_interp, cpu)
