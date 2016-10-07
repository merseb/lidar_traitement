# -*- coding: utf-8 -*-

import numpy as np
from netCDF4 import Dataset, date2index
from joblib import Parallel, delayed
import time
from datetime import datetime
import os
from glob import glob



def splitlist(a, n):
    """
    retourne un iterateur pour diviser une liste a en n sous-listes +- egales
    """
    k, m = len(a) / n, len(a) % n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))


def rolling_window(matrice, longitudes, latitudes, coords, window, reso):
    reso_init = np.abs(np.round(np.mean(np.diff(latitudes)), 2))
    list_stats = []
    for coord in coords:
        stats = np.zeros(5)
        stats[:] = np.nan
        x_px = coord[0]
        y_px = coord[1]
        x_min = x_px - (reso * (window / 2))
        x_max = x_px + (reso * (window / 2))
        y_min = y_px - (reso * (window / 2))
        y_max = y_px + (reso * (window / 2))
    
        if x_min > longitudes[-1] or x_max < longitudes[0] or y_min < latitudes[-1] or y_max > latitudes[0]:
            list_stats.append(stats)
        else:
            if x_min >= longitudes[0]:
                j0 = np.int(np.abs(longitudes[0] - x_min) / reso_init)
            else:
                j0 = 0
            if x_max <= longitudes[-1]:
                j1 = np.int(np.abs(longitudes[0] - x_max) / reso_init)
            else:
                j1 = longitudes.shape[0] + 1
    
            if y_min > latitudes[-1]:
                i1 = np.int(np.abs(latitudes[0] - y_min) / reso_init)
            else:
                i1 = latitudes.shape[0]+1
            if y_max < latitudes[0]:
                i0 = np.int(np.abs(latitudes[0] - y_max) / reso_init)
            else:
                i0 = 0
            m = matrice[i0:i1, j0:j1]
            if m.size:
                stats[0] = 100 * (np.count_nonzero(~np.isnan(m)) / float(m.shape[0] * m.shape[1]))
                if stats[0] != 0:
                    stats[1] = np.nanmean(m)
                    stats[2] = np.nanmin(m)
                    stats[3] = np.nanmax(m)
                    stats[4] = np.nanstd(m)
            list_stats.append(stats)
    return np.asarray(list_stats)



path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
tmpdir = os.getcwd()

file_nc = 'seviri_r005_16d.nc'
var = 'AOD_VIS06'
cpu = 3
date = datetime(2014,01,27)
window = 9

##### liste de coordonnees
x_min, x_max = -25.0, 57.01
y_min, y_max = -1.25, 51.01
reso = 0.25
xo = np.arange(x_min, x_max, reso)  #longitudes du .nc
yo = np.arange(y_min, y_max, reso)[::-1]  #latitudes du .nc
xx, yy = np.meshgrid(xo, yo) # produit cartesien des lon/lat
xy = zip(xx.flatten(), yy.flatten()) # liste de tuples(lon/lat)

##### repartition des couples lat/lon dans n sous-listes = nombre de processeurs pour la parallelisation
list_jobs = [job for job in splitlist(xy, cpu)] 

t1 = time.time()
nc = Dataset(path+'/donnees_annexes/'+file_nc)
t1 = time.time()
ncdates = nc.variables['time']
id_date = date2index(date, ncdates, calendar=ncdates.calendar, select='after') # def de l'index de la date traitee
mat = np.ma.filled(nc.variables[var][id_date, ...], np.nan)  # variable 3d: t,y,x
lg = nc.variables['longitude'][:]
lt = nc.variables['latitude'][:]
nc.close()
##### parallelisation
values = rolling_window(mat, lg, lt, xy, window, reso)
t2 = time.time() - t1
print(t2, ' sec')
arr = np.vstack(values)  # empilement des matrices en sortie
print('done')