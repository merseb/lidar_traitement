# -*- coding: utf-8 -*-

import numpy as np
from netCDF4 import Dataset, date2index
from joblib import Parallel, delayed
import time


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
    vmoy = np.average(valeurs, axis=0, weights=weight)
    vmin = np.min(valeurs.T * weight, axis=1)
    vmax = np.max(valeurs.T * weight, axis=1)
    variance = np.average((valeurs-vmoy)**2, axis=0, weights=weight)
    vstd = np.sqrt(variance)
    return np.append(valeurs.shape[0],(vmoy, vmin, vmax, vstd))


def ponderation(ponderation_type, lidar_parametres, list_df, coords, window, reso):
    """
    Fonction qui retourne pour les pixels de coordonnees(coords) la valeur ponderee
    """
    v = []
    app = v.append
    
    for coord in coords[:]:
        x_px = coord[0]
        y_px = coord[1]
        # definition de la fenetre
        x_min = x_px - reso*(window/2)
        x_max = x_px + reso*(window/2)
        y_min = y_px - reso*(window/2)
        y_max = y_px + reso*(window/2)

        # pour chaque sous-categorie:
        for df in list_df:
            arr = np.asarray([])
            longitude = df.Longitude.values
            latitude = df.Latitude.values
            ##### recherche des indices des valeurs de lat/lon comprises dans le "pixel" de coord (x_px,y_px)
            idx = np.where((latitude >= y_px - 0.25) & (latitude < y_px) & (longitude >= x_px) & (longitude < x_px + 0.25))[0]
            
            if idx.size:
                arr = np.append(arr, df[lidar_parametres].ix[idx].mean().values)
            else:
                arr = np.append(arr, np.asarray([np.nan]*len(lidar_parametres)))
            ##### recherche des indices de valeurs lidar dans la fenetre
            idx_lidar = list(np.where((latitude >= y_min) & (latitude < y_max) & (longitude >= x_min) & (longitude < x_max))[0])
            dist = np.asarray([distance(longitude[i], latitude[i], x_px, y_px) for i in idx_lidar])
            app(arr)
            ##### calcul de la distance de chaque point lidar au pixel(x_px,y_px) pour chaque parametre lidar
            if len(idx_lidar):
                if ponderation_type == 'carredistance':            
                    weight = (1/dist**2)/np.sum(1/dist**2)
                else:
                    weight = (1/dist)/np.sum(1/dist)
                arr2 = np.append(arr2, calcPonderation(weight, df[lidar_parametres].ix[idx_lidar].values))
            else:
                arr_nan = np.zeros((1 + len(lidar_parametres)*4))
                arr_nan[:] = np.nan
                arr2 = np.append(arr2, arr_nan)
            app(arr2)
    #### retourne pour chaque sous-categorie array([nb_pixel, vmoy, vmin, vmax, vstd])
    return [np.vstack([v[i] for i in range(0,len(v), len(list_df))]), np.vstack([v[i] for i in range(1,len(v), len(list_df))])]

def ponderation2(ponderation_type, lidar_parametres, lidar_df, subtypes, coords, window, reso):
    """
    Fonction qui retourne pour les pixels de coordonnees(coords) la valeur ponderee
    """
    
    longitude = lidar_df.Longitude.values
    latitude = lidar_df.Latitude.values

    v = []
    app = v.append
    arr = np.asarray([])
    arr2 = np.asarray([])
    for coord in coords[857:858]: #[850:860]:
        x_px = coord[0]
        y_px = coord[1]
        # definition de la fenetre
        x_min = x_px - reso*(window/2)
        x_max = x_px + reso*(window/2)
        y_min = y_px - reso*(window/2)
        y_max = y_px + reso*(window/2)
        ##### recherche des indices des valeurs lidar comprises dans le "pixel" de coord (x_px,y_px)
        idx_px = np.where((latitude >= y_px - 0.25) & (latitude < y_px) & (longitude >= x_px) & (longitude < x_px + 0.25))[0]
        ##### recherche des indices de valeurs lidar dans la fenetre
        idx_wdw = list(np.where((latitude >= y_min) & (latitude < y_max) & (longitude >= x_min) & (longitude < x_max))[0])
        if (idx_px.size) or (len(idx_wdw)):
            for subtype in subtypes:
                df_subT = lidar_df[lidar_df.FeatureSubtype == subtype]
                if idx_px.size:
                    arr = np.append(arr, df_subT[lidar_parametres].ix[idx_px][subtype].mean().values)
                else:
                    arr = np.append(arr, np.asarray([np.nan]*len(lidar_parametres)))
                app(arr)
                ##### calcul de la distance de chaque point lidar au pixel(x_px,y_px) pour chaque parametre lidar
                if len(idx_wdw):
                    subT_lon = df_subT.Longitude
                    subT_lat = df_subT.Latitude
                    dist = np.asarray([distance(subT_lon.ix[i], subT_lat.ix[i], x_px, y_px) for i in idx_wdw])
                    if ponderation_type == 'carredistance':            
                        weight = (1/dist**2)/np.sum(1/dist**2)
                    else:
                        weight = (1/dist)/np.sum(1/dist)
                    arr2 = np.append(arr2, calcPonderation(weight, df_subT[lidar_parametres].ix[idx_wdw].values))
                else:
                    arr_nan = np.zeros((1 + len(lidar_parametres)*4))
                    arr_nan[:] = np.nan
                    arr2 = np.append(arr2, arr_nan)
                app(arr2)
    #### retourne pour chaque sous-categorie array([nb_pixel, vmoy, vmin, vmax, vstd])
    return [np.vstack([v[i] for i in range(0,len(v), len(list_df))]), np.vstack([v[i] for i in range(1,len(v), len(list_df))])]



def points2grid(coord_px, lidar_df, lidar_parametres, subtypes):
    """
    Convertit une liste de points en matrice
    
    PARAMETRES:
    
    **coord_px** (*liste,tuple*): coordonnees x,y du pixel \n
    **longitude** (*list*): liste des longitudes a traiter \n
    **latitude** (*list*): liste des latitudes a traiter \n
    **valeurs** (*list*): liste des valeurs \n

    Retourne en (x,y) la moyenne des valeurs 
    """

    x = coord_px[0]
    y = coord_px[1]
    latitude = lidar_df.Latitude.values
    longitude = lidar_df.Longitude.values
    idx = np.where((latitude >= y - 0.25) & (latitude < y) & (longitude >= x) & (longitude < x + 0.25))[0]  # recherche des indices des valeurs de lat/lon comprises dans le "pixel" de coord (x[j],y[i])
    arr = np.asarray([])
    if idx.size:
        for subtype in subtypes:
            arr = np.append(arr, lidar_df[lidar_df.FeatureSubtype == subtype][lidar_parametres].ix[idx].mean().values)
    else:
        arr = np.append(arr, np.asarray([np.nan]*len(lidar_parametres)*len(subtypes)))
    return arr

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

    print 'lidar'
    
    ##### boucle sur la liste des sous-types 
    list_values = {} # initialisation du dictionnaire
    subtypes = list(lidar_df.FeatureSubtype.unique())  # liste des sous-categories
    list_df = [lidar_df[lidar_df.FeatureSubtype == st].reset_index(drop=True) for st in subtypes]  # liste des df par sous-categorie
    t1 = time.time()
    grid = np.vstack([points2grid(coords, lidar_df, lidar_parametres, subtypes) for coords in xy])
    print time.time()-t1
    interp_values = Parallel(n_jobs=cpu)(delayed(ponderation)(ponderation_type, lidar_parametres, list_df, lcoords, window, reso) for lcoords in list_jobs)
    ##### 'reconstruction' et chargement des matrices dans le dictionnaire
    for s in range(len(subtypes)):
        m = np.vstack((interp_values[i][s] for i in range(cpu)))
        list_values[subtypes[s]+'_nb_lidar_points'] = m[:, 0]
        for p in range(len(lidar_parametres)):
            list_values[subtypes[s]+'_'+lidar_parametres[p]] = np.vstack((m[:, p + i] for i in range(1, m.shape[1] - 1, len(lidar_parametres)))).T
    print('%s sec' % str(time.time() - t1))
    #####

    ##### boucle sur la liste des fichiers externes f est une liste comprenant l'adresse du fichier et le nom de la variable
    for f in ext_files:
        print f[1]
        t1 = time.time()
        try:
            nc = Dataset(f[0], 'r')
        except IOError:
            print 'fichier impossible a ouvrir'
            pass
        ncdates = nc.variables['time']
        id_date = date2index(date, ncdates, calendar=ncdates.calendar, select='after') # def de l'index de la date traitee
        if f[1] == 'mean_pDUST':
            mat = np.ma.filled(nc.variables[f[1]][id_date,0, ...], np.nan)  # variable 4d: t,z,y,x
        else:
            mat = np.ma.filled(nc.variables[f[1]][id_date, ...], np.nan)  # variable 3d: t,y,x
        lg = nc.variables['longitude'][:]
        lt = nc.variables['latitude'][:]
        nc.close()
        ##### parallelisation
        values = Parallel(n_jobs=cpu)(delayed(rolling_window)(mat, lg, lt, lcoords, window, reso) for lcoords in list_jobs)
        print('%s sec' % str(time.time() - t1))
        list_values[f[1]] = np.vstack(values)  # empilement des matrices en sortie
    #####
    return list_values
