# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from netCDF4 import Dataset, date2index
from joblib import Parallel, delayed
import time
import warnings
import matplotlib.pyplot as plt

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





cpu = 3
window = 9
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

# liste des variables/parametres interpolees pour chaque sous-type
lidar_parametres = ["Column_Optical_Depth_Aerosols_532", "Feature_Optical_Depth_532", "Feature_Optical_Depth_Uncertainty_532", "Top_corr", "Base_corr", "Concentration_Aerosols", "Relative_Humidity"]
ponderation_type = 'carredistance' #'distance', 'carredistance'
subtypes = ['dust', 'polluted_dust'] # sous-types d'aerosols extraits des donnees lidar
w_interp = 9 # 3 6 9 12 15 ... fenetre glissante pour l'interpolation

lidar_df = pd.read_csv("/home/mers/code/python/lidar_traitement/out/2014_01_27_16d.csv",header=0)
matrices = [ lidar_df[lidar_df.columns[1:]][lidar_df.FeatureSubtype == subtype].values for subtype in subtypes]


t1 = time.time()
subsets = {}
for s in range(len(subtypes)):
    interp_values = Parallel(n_jobs=cpu)(delayed(ponderation)(matrices[s], lcoords, ponderation_type, reso, window) for lcoords in list_jobs)
    subsets[subtypes[s]] = np.vstack(interp_values)
print("%.2f sec" % (time.time()-t1))





### indices 22320:22330 ---> 22325 pour indice_px et indice_wdw
### indices 22509:22513 ---> indice_wdw (1 seule valeur)
matrice = matrices[0]
indices_px = [calcIndices(coord, matrice[:,0], matrice[:,1], reso) for coord in coords[:]] # liste des indices des valeurs lidar pour chaque pixel du raster de sortie
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    moyennes = np.vstack([np.nanmean(matrice[ind,2:] , axis=0) for ind in indices_px])  # calcul de la moyenne des valeurs lidar pour chaque pixel du raster de sortie
indices_wdw = [calcIndices(coord, matrice[:,0], matrice[:,1], reso, window) for coord in coords[:]] # liste des indices des valeurs lidar dans la fenetre autour de chaque pixel du raster de sortie
lidx = np.array([len(ind) for ind in indices_wdw])  # tableau listant le nombre d'indices pour chaque pixel du raster en sortie
lidx1 = np.vstack([(ix, len(ind)) for ix, ind in enumerate(indices_wdw)])
mpxs = [np.repeat(np.asarray(coords[i])[np.newaxis],lidx[i], axis=0) for i in range(lidx.shape[0])]  # liste de tableaux, chacun ayant un nombre de lignes = au nb d'indices dans la fenetre considérée et nb colonnes= 2 (couple lonlat ou coord du pixel du raster)
lonlats = [(matrice[j,:2], mpxs[i]) for i,j in enumerate(indices_wdw) if j.size] # liste de tableaux, chacun 
lonlats1 = [np.column_stack((matrice[j,:2], mpxs[i])) for i,j in enumerate(indices_wdw) if j.size]
mats = [matrice[ind,2:] for ind in indices_wdw if ind.size]
dists = [np.sqrt(np.sum((lonlats[i][0] - lonlats[i][1])**2, axis=1).astype(np.float)) for i in range(len(lonlats))]
dists1 = [np.sqrt(np.sum((lonlats1[i][:,:2] - lonlats1[i][:,2:])**2, axis=1).astype(np.float)) for i in range(len(lonlats))]
if ponderation == 'carredistance':            
    weights = [(1/dist**2)/np.sum(1/dist**2) for dist in dists]
else:
    weightsb = [(1/dist)/np.sum(1/dist) for dist in dists]
pondere = [calcPonderation(weights[i], mats[i]) for i in range(len(lonlats))]
lid = np.where(lidx > 0)[0]
arr = np.zeros((len(indices_wdw), 25))
arr[:] = np.nan
arr[lid,...] = pondere

plt.imshow(subsets['dust'][:,1].reshape(yo.shape[0], -1)),plt.colorbar()
