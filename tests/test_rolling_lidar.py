# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2index
from joblib import Parallel, delayed
import time
from datetime import datetime


def splitlist(a, n):
    """
    retourne un iterateur pour diviser une liste a en n sous-listes +- egales
    """
    k, m = len(a) / n, len(a) % n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))


def distance(xa, ya, xb, yb):
    """
    """
    return np.sqrt((xb-xa)**2 + (yb-ya)**2)


def calcPonderation(weight, valeurs):
    """
    """
    vmoy = np.average(valeurs, weights=weight, axis=0)
    vmin = np.min(valeurs.T * weight, axis=1)
    vmax = np.max(valeurs.T * weight, axis=1)
    variance = np.average((valeurs-vmoy)**2, axis=0, weights=weight)
    vstd = np.sqrt(variance)
    return np.append(valeurs.shape[0], (vmoy, vmin, vmax, vstd))


def calcIndices(coordonnees, longitude, latitude, resolution, window=1):
    x_px = coordonnees[0]
    y_px = coordonnees[1]
    if window == 1 :
        indices = np.where((latitude >= y_px - resolution) & (latitude < y_px) & (longitude >= x_px) & (longitude < x_px + resolution))[0]
    else:
        # definition de la fenetre
        x_min = x_px - resolution*(window/2)
        x_max = x_px + resolution*(window/2)
        y_min = y_px - resolution*(window/2)
        y_max = y_px + resolution*(window/2)
        indices = np.where((latitude >= y_min) & (latitude < y_max) & (longitude >= x_min) & (longitude < x_max))[0]
    return indices

def ponderation(matrice, coords, ponderation, resolution, window):
    """
    """
    nbstats = (matrice.shape[1]-2)*4 + 1
    indices_px = [calcIndices(coord, matrice[:,0], matrice[:,1], resolution) for coord in coords[:]]
    moyennes = np.vstack([np.nanmean(matrice[ind,2:] , axis=0) for ind in indices_px])
    indices_wdw = [calcIndices(coord, matrice[:,0], matrice[:,1], resolution, window) for coord in coords[:]]
    lidx = np.array([len(ind) for ind in indices_wdw])
    mpxs = [np.repeat(np.asarray(coords[i])[np.newaxis],lidx[i], axis=0) for i in range(lidx.shape[0])]
    lonlats = [(matrice[j,:2], mpxs[i]) for i,j in enumerate(indices_wdw) if j.size]
    mats = [matrice[ind,2:] for ind in indices_wdw if ind.size]
    dists = [np.sqrt(np.sum((lonlats[i][0] - lonlats[i][1])**2, axis=1).astype(np.float)) for i in range(len(lonlats))]
    if ponderation == 'carredistance':            
        weights = [(1/dist**2)/np.sum(1/dist**2) for dist in dists if dist.size]
    else:
        weights = [(1/dist)/np.sum(1/dist) for dist in dists]
    pondere = [calcPonderation(weights[i], mats[i]) for i in range(len(lonlats))]
    lid = np.where(lidx > 0)[0]
    arr = np.zeros((len(indices_wdw), nbstats))
    arr[:] = np.nan
    arr[lid,...] = pondere
    return np.column_stack((moyennes, arr))



def rolling_window(matrice, longitudes, latitudes, coords, window, reso):
    reso_init = np.abs(np.round(np.mean(np.diff(latitudes)), 2))
    list_stats = []
    app = list_stats.append
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
            app(stats)
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
            app(stats)
    return np.asarray(list_stats)



def extractData(ponderation_type, lidar_parametres, lidar_df, ext_files, date, window, cpu, x, y, reso):
    """
    Fonction intermediaire qui appelle la fonction de ponderation et la fonction d'extraction des donnees externes.
    Elle retourne un dictionnaire comprenant une entree pour chaque variable de chaque type d'aerosol ainsi que pour
    chaque variable externe.
    A chaque entree lidar correspond une liste comprenant d'une part une matrice(N x 5) ou N est le nombre de couples xy et 5 
    les calculs : nombre de valeurs lidar, valeur interpolee, min, max, std; d'autre part une sous-liste par couple de
    xy regroupant les valeurs lidar exploitees.
    A chaque entree concernant les donnees externes correspond une matrice(N x 5) ou N est le nombre de couples xy et 5
    les calculs de : pourcentage de de pixels non nuls, moyenne, min, max, std.

    PARAMETRES:

    **ponderation_type**(*string*): type de ponderation appliqué pour l'interpolation des donnees lidar
    **lidar_parametres**(*list*): liste des variables lidar interpolees transmises a la fonction de ponderation
    **lidar_df**(*pandas dataframe object*): dataframe regroupant l'ensemble des donnees filtrees pour l'ensemble des sous-types
    **ext_files**(*list*): liste de listes regroupant le chemin et le nom de la variable pour chaque donnee externe
    **date** (*datetetime object*): date
    **window** (*int*): largeur de la fenetre glissante
    **cpu** (*int*): nombre de processeurs pour la parallelisation
    **x** (*list(float)*): liste des longitudes en sortie
    **y** (*list(float)*): liste des latitudes en sortie
    **reso** (*float*): resolution spatiale

    """
    ##### liste de coordonnees
    xx, yy = np.meshgrid(x, y) # produit cartesien des lon/lat
    xy = zip(xx.flatten(), yy.flatten()) # liste de tuples(lon/lat)
    ##### repartition des couples lat/lon dans n sous-listes = nombre de processeurs pour la parallelisation
    list_jobs = [job for job in splitlist(xy, cpu)]
    subtypes = list(lidar_df.FeatureSubtype.unique())  # liste des sous-categories
    matrices = [lidar_df[lidar_df.columns[1:]][lidar_df.FeatureSubtype == subtype].values for subtype in subtypes]  # liste des df par sous-categorie
    
    ##### boucle sur la liste des sous-types 
    print 'lidar'
    t1 = time.time()
    subsets = {} # initialisation du dictionnaire
    for s in range(len(subtypes)):
        interp_values = Parallel(n_jobs=cpu)(delayed(ponderation)(matrices[s], lcoords, ponderation_type, reso, window) for lcoords in list_jobs)
        subsets[subtypes[s]] = np.vstack(interp_values)
    #####
    print('%s sec' % str(time.time() - t1))

#    ##### boucle sur la liste des fichiers externes f est une liste comprenant l'adresse du fichier et le nom de la variable
#    for f in ext_files:
#        print f[1]
#        t1 = time.time()
#        try:
#            nc = Dataset(f[0], 'r')
#            ncdates = nc.variables['time']
#            id_date = date2index(date, ncdates, calendar=ncdates.calendar, select='after') # def de l'index de la date traitee
#            if f[1] == 'mean_pDUST':
#                mat = np.ma.filled(nc.variables[f[1]][id_date,0, ...], np.nan)  # variable 4d: t,z,y,x
#            else:
#                mat = np.ma.filled(nc.variables[f[1]][id_date, ...], np.nan)  # variable 3d: t,y,x
#            lg = nc.variables['longitude'][:]
#            lt = nc.variables['latitude'][:]
#            nc.close()
#        except IOError:
#            print 'fichier impossible a ouvrir'
#            pass
#        ##### parallelisation
#        values = Parallel(n_jobs=cpu)(delayed(rolling_window)(mat, lg, lt, lcoords, window, reso) for lcoords in list_jobs)
#        print('%s sec' % str(time.time() - t1))
#        subsets[f[1]] = np.vstack(values)  # empilement des matrices en sortie
    #####
    return subsets


cpu = 3
window = 9
date = datetime(2007,01,27)
# zone d'etude
x_min, x_max = -25.0, 57.01
y_min, y_max = -1.25, 51.01
reso = 0.25
xo = np.arange(x_min, x_max, reso)  #longitudes du .nc
yo = np.arange(y_min, y_max, reso)[::-1]  #latitudes du .nc
xxo, yyo = np.meshgrid(xo, yo) # produit cartesien des lon/lat
xyo = zip(xxo.flatten(), yyo.flatten()) # liste de tuples(lon/lat)
list_jobs = [job for job in splitlist(xyo, cpu)]
coords = list_jobs[1]



###### liste des donnees externes integrees au fichier netcdf en sortie
path_ext = '/home/mers/code/python/lidar_traitement/donnees_annexes/'
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
fichiers_ext = [[fnc1, var1], [fnc2, var2], [fnc3, var3], [fnc4, var4]]
var_ext = [v[1] for v in fichiers_ext]

# liste des variables/parametres interpolees pour chaque sous-type
lidar_parametres = ["Column_Optical_Depth_Aerosols_532", "Feature_Optical_Depth_532", "Feature_Optical_Depth_Uncertainty_532", "Top_corr", "Base_corr", "Concentration_Aerosols", "Relative_Humidity"]
ponderation_type = 'carredistance' #'distance', 'carredistance'
subtypes = ['dust', 'polluted_dust'] # sous-types d'aerosols extraits des donnees lidar
w_interp = 9 # 3 6 9 12 15 ... fenetre glissante pour l'interpolation

lidar_df = pd.read_csv("/home/mers/code/python/lidar_traitement/out/2014_01_27_16d_test2.csv",header=0)

list_values = {}
t1 = time.time()

output = extractData(ponderation_type, lidar_parametres, lidar_df, fichiers_ext, date, window, cpu, xo, yo, reso)

t2 = time.time() - t1
print('%.3f sec' %t2)


import matplotlib.pyplot as plt
plt.imshow(output['dust'][:,6].reshape(yo.shape[0],-1)),plt.colorbar()