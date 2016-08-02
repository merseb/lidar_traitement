# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2index
import os
from glob import glob
from joblib import Parallel, delayed
import time
from itertools import chain
from collections import deque
from datetime import datetime


def splitlist(a, n):
    """
    retourne un iterateur pour diviser une liste a en n sous-listes +- egales
    """
    k, m = len(a) / n, len(a) % n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))


#def ia(gen, fill=0, size=2, fill_left=True, fill_right=False):
#    gen, ssize = iter(gen), size - 1
#    deq = deque(chain([fill] * ssize * fill_left, (next(gen) for _ in xrange((not fill_left) * ssize))), maxlen = size)
#    for item in chain(gen, [fill] * ssize * fill_right):
#        deq.append(item)
#        yield deq
#
#
#def rolling_window(a, window):
#    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
#    strides = a.strides + (a.strides[-1],)
#    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def write_line(fichiers_keys, lidar_keys, data):
    line = []
    for lp in lidar_keys:
        line += ['interp_'+lp]
        line += ['  ']
        line += [data[lp][0].tolist()]
        line += ['  ']
        line += ['lidar_'+lp]
        try:
            line += data[lp][1]
        except ValueError:
            line += data[lp][1].tolist()
    for f in fichiers_ext:
        line += ['  ///  ']
        line += [f[1]]
        line += ['  ']
        line += data[f[1]].tolist()
    return line


def exportTXT(fichiers_ext, valeurs, date, prefix):
    lidar_param = sorted(list(set(valeurs.keys()) - set([f[1] for f in fichiers_ext])))
    with open('/home/mers/Bureau/teledm/donnees/lidar/calipso/caliop/out/'+str(date.date()).replace('-', '_')+'_info_interp_'+prefix+'.txt', 'w') as info:
        for i in range(len(valeurs[lidar_param[0]][0])):
            tmp = []
            for lp in lidar_param:
                tmp += ['interp_'+lp]
                tmp += ['  ']
                tmp += [valeurs[lp][0][i].tolist()]
                tmp += ['  ']
                tmp += ['lidar_'+lp]
                try:
                    tmp += valeurs[lp][1][i]
                except ValueError:
                    tmp += valeurs[lp][1][i].tolist()
            for f in fichiers_ext:
                tmp += ['  ///  ']
                tmp += [f[1]]
                tmp += ['  ']
                tmp += valeurs[f[1]][i].tolist()
            info.write(str(tmp)+'\n')


#def carreDistance(xa, ya, xb, yb, val_xaya):
#    """
#    Pour chaque pixel(xb,yb)
#    la fonction retourne l'inverse de la distance au point lidar(xa,ya) elevee au carre
#    ainsi que la valeur(lidar) modifiee: p_val = valxaya/distance au carre
#    """
#    dist_carre = (xb-xa)**2 + (yb-ya)**2
#    p_val = val_xaya / dist_carre
#    return 1/dist_carre, p_val


def distance(xa, ya, xb, yb):
    return np.sqrt((xb-xa)**2 + (yb-ya)**2)
    

def calcPonderation(weight, valeurs):
    vmoy = np.average(valeurs, weights=weight)
    vmin = np.min(valeurs * weight)
    vmax = np.max(valeurs * weight)
    variance = np.average((valeurs-vmoy)**2, weights=weight)
    vstd = np.sqrt(variance)
    return [np.asarray([valeurs.shape[0], vmoy, vmin, vmax, vstd]), valeurs]


def ponderation1(ponderation_type, lidar_parametres, lidar_df, coord, window, reso):
    """
    Fonction qui retourne pour le pixel de coordonnees(coords) la valeur ponderee
    ainsi que la liste/valeur des points lidar utilises(idx_lidar)
    """
    longitude = lidar_df.Longitude.values
    latitude = lidar_df.Latitude.values
    x_px = coord[0]
    y_px = coord[1]
    # definition de la fenetre
    x_min = x_px - reso*(window/2)
    x_max = x_px + reso*(window/2)
    y_min = y_px - reso*(window/2)
    y_max = y_px + reso*(window/2)

    ##### recherche des valeurs dans la fenetre
    idx_lidar = list(np.where((latitude >= y_min) & (latitude < y_max) & (longitude >= x_min) & (longitude < x_max))[0])
    dist = np.asarray([distance(longitude[i], latitude[i], x_px, y_px) for i in idx_lidar])
    ##### calcul de la distance de chaque point lidar au pixel(x_px,y_px) pour chaque parametre lidar
    if len(idx_lidar):
        if ponderation_type == 'CarreDistance':            
            weight = (1/dist**2)/np.sum(1/dist**2)
        else:
            weight = (1/dist)/np.sum(1/dist)
        return [calcPonderation(weight, lidar_df[lp].ix[idx_lidar].values) for lp in lidar_parametres]
    else:
        return [[np.array([np.nan]*5),[np.nan]]]*len(lidar_parametres)


def ponderation(ponderation_type, lidar_parametres, list_df, coord, window, reso):
    """
    Fonction qui retourne pour le pixel de coordonnees(coords) la valeur ponderee
    ainsi que la liste/valeur des points lidar utilises(idx_lidar)
    """
    x_px = coord[0]
    y_px = coord[1]
    # definition de la fenetre
    x_min = x_px - reso*(window/2)
    x_max = x_px + reso*(window/2)
    y_min = y_px - reso*(window/2)
    y_max = y_px + reso*(window/2)
    v_out = []
    for df in list_df:
        longitude = df.Longitude.values
        latitude = df.Latitude.values
        ##### recherche des valeurs dans la fenetre
        idx_lidar = list(np.where((latitude >= y_min) & (latitude < y_max) & (longitude >= x_min) & (longitude < x_max))[0])
        dist = np.asarray([distance(longitude[i], latitude[i], x_px, y_px) for i in idx_lidar])
        ##### calcul de la distance de chaque point lidar au pixel(x_px,y_px) pour chaque parametre lidar
        if len(idx_lidar):
            if ponderation_type == 'CarreDistance':            
                weight = (1/dist**2)/np.sum(1/dist**2)
            else:
                weight = (1/dist)/np.sum(1/dist)
            v_out.append([calcPonderation(weight, df[lp].ix[idx_lidar].values) for lp in lidar_parametres])
        else:
            v_out.append([[np.array([np.nan]*5),[np.nan]]]*len(lidar_parametres))
    return v_out


def ponderation_helper(ponderation_type, lidar_parametres, list_df, coords, window, reso):
    """
    Fonction qui permet d'envoyer la fonction ponderation sous forme de list comprehension
    """
    return [ponderation(ponderation_type, lidar_parametres,list_df, px, window, reso) for px in coords]


#def extractData(matrice, longitude, latitude, coords, window, reso=0.25):
#    """
#    Fonction qui extrait les pourcentage de pixels non nuls, moyenne, min, max, std a partir d'une fenetre glissante
#    qui est definie par les latitudes/longitudes.
#    Elle retourne une matrice(n,5) n = nombre de tuples lon/lat dans coords  
#    
#    PARAMETRES:
#    
#    **matrice**(*1d*): matrice issue des donnees externes de longueur NxM
#    **longitude**(*matrice 1dim NxM*)
#    **latitude**(*matrice 1dim NxM*)
#    **coords** (*liste*): liste de coordonnees provenant de la grille de sortie
#    **reso** (*float*): resolution spatiale de la grille de sortie
#    """
#    data_values = []
#    ##### pour chaque tuple lon/lat
#    for coord in coords:
#        tmp = np.zeros(5, float)
#        tmp[:] = np.nan
#        x_px = coord[0]
#        y_px = coord[1]
#        # definition de la fenetre
#        x_min = x_px - reso*(window/2)
#        x_max = x_px + reso*(window/2)
#        y_min = y_px - reso*(window/2)
#        y_max = y_px + reso*(window/2)
#        ##### extraction des indices compris dans la fenetre
#        idx_prod = list(np.where((latitude >= y_min) & (latitude < y_max) & (longitude >= x_min) & (longitude < x_max))[0])
#        mat_valeurs = matrice[idx_prod]
#        nbpx = mat_valeurs.shape[0]
#        nbpx_nonull = np.count_nonzero(~np.isnan(mat_valeurs))
#        try:
#            tmp[0] = 100*(nbpx_nonull/nbpx)  # pourcentage de pixels non nuls
#            if nbpx_nonull > 0:
#                tmp[1] = np.nanmean(mat_valeurs)  # valeur moyenne
#                tmp[2] = np.nanmin(mat_valeurs)  # valeur min
#                tmp[3] = np.nanmax(mat_valeurs)  # valeur max
#                tmp[4] = np.nanstd(mat_valeurs)  # ecart-type
#        except ZeroDivisionError:
#            tmp[:] = np.nan
#        data_values.append(tmp)
#    return np.asarray(data_values)
#
#
#def extractData2(matrice, longitude, latitude, coord, window, reso):
#    """
#    Fonction qui extrait les pourcentage de pixels non nuls, moyenne, min, max, std a partir d'une fenetre glissante
#    qui est definie par les latitudes/longitudes.
#    Elle retourne une matrice(n,5) n = nombre de tuples lon/lat dans coords  
#    
#    PARAMETRES:
#    
#    **matrice**(*1d*): matrice issue des donnees externes de longueur NxM
#    **longitude**(*matrice 1dim NxM*)
#    **latitude**(*matrice 1dim NxM*)
#    **coords** (*liste*): liste de coordonnees provenant de la grille de sortie
#    **reso** (*float*): resolution spatiale de la grille de sortie
#    """
#    ##### pour chaque tuple lon/lat
#    x_px = coord[0]
#    y_px = coord[1]
#    # definition de la fenetre
#    x_min = x_px - reso*(window/2)
#    x_max = x_px + reso*(window/2)
#    y_min = y_px - reso*(window/2)
#    y_max = y_px + reso*(window/2)
#    ##### extraction des indices compris dans la fenetre
#    tmp = np.zeros(5, float)
#    tmp[:] = np.nan
#    idx_prod = list(np.where((longitude >= y_min) & (longitude < y_max) & (latitude >= x_min) & (latitude < x_max))[0])
#    mat_valeurs = matrice[idx_prod]
#    nbpx = mat_valeurs.shape[0]
#    nbpx_nonull = np.count_nonzero(~np.isnan(mat_valeurs))
#    try:
#        tmp[0] = 100*(nbpx_nonull/nbpx)  # pourcentage de pixels non nuls
#        if nbpx_nonull > 0:
#            tmp[1] = np.nanmean(mat_valeurs)  # valeur moyenne
#            tmp[2] = np.nanmin(mat_valeurs)  # valeur min
#            tmp[3] = np.nanmax(mat_valeurs)  # valeur max
#            tmp[4] = np.nanstd(mat_valeurs)  # ecart-type
#    except ZeroDivisionError:
#        tmp[:] = np.nan
#    return tmp  #np.asarray(data_values)


def extractData_helper(matrice, longitude, latitude, coords, window, reso):
    return [rolling_window(matrice, longitude, latitude, coord, window, reso) for coord in coords]


def rolling_window(matrice, longitudes, latitudes, coord, window, reso):
    reso_init = np.abs(np.round(np.mean(np.diff(latitudes)), 2))
    stats = np.zeros(5)
    stats[:] = np.nan
    x_px = coord[0]
    y_px = coord[1]
    x_min = x_px - (reso * (window / 2))
    x_max = x_px + (reso * (window / 2))
    y_min = y_px - (reso * (window / 2))
    y_max = y_px + (reso * (window / 2))

    if x_min > longitudes[-1] or x_max < longitudes[0] or y_min < latitudes[-1] or y_max > latitudes[0]:
        return stats
    else:
        if x_min >= longitudes[0]:
            j0 = np.int(np.abs(longitudes[0] - x_min) / reso_init)
        else: j0 = 0
        if x_max <= longitudes[-1]:
            j1 = np.int(np.abs(longitudes[0] - x_max) / reso_init)
        else: j1 = longitudes.shape[0] + 1        
    
        if y_min > latitudes[-1]:
            i1 = np.int(np.abs(latitudes[0] - y_min) / reso_init)
        else: i1 = latitudes.shape[0]+1
        if y_max < latitudes[0]:
            i0 = np.int(np.abs(latitudes[0] - y_max) / reso_init)
        else: i0 = 0
        m = matrice[i0:i1,j0:j1]
        if m.size:
            stats[0] = 100 * (np.count_nonzero(~np.isnan(m)) / float(m.shape[0] * m.shape[1]))
            if stats[0] != 0:
                stats[1] = np.nanmean(m)
                stats[2] = np.nanmin(m)
                stats[3] = np.nanmax(m)
                stats[4] = np.nanstd(m)
        return stats


def extractData3(ponderation_type, lidar_parametres, lidar_stype, lidar_df, ext_files, date, window, cpu, x, y, reso):
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
    **lidar_stype**(*list*): liste des types d'aerosol traites
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
    subtypes = list(lidar_df.FeatureSubtype.unique())
    list_df = [ lidar_df[lidar_df.FeatureSubtype == st].reset_index(drop=True) for st in subtypes]
    t1 = time.time()
    interp_values = Parallel(n_jobs=cpu)(delayed(ponderation_helper)(ponderation_type, lidar_parametres, list_df, lcoords, window, reso) for lcoords in list_jobs)
    ##### 'reconstruction' et chargement des matrices et listes dans le dictionnaire
    tmp = []
    for i in range(cpu):
        tmp += interp_values[i]
    for st in range(len(subtypes)):
        for p in range(len(lidar_parametres)):
            list_values[subtypes[st]+'_'+lidar_parametres[p]] = [np.vstack([tmp[i][st][p][0] for i in range(len(xy))]), [tmp[i][st][p][1] for i in range(len(xy))]]
    t2 = time.time() - t1
    print t2, ' sec'
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
            mat = np.ma.filled(nc.variables[f[1]][id_date,0, ...], np.nan)
        else:
            mat = np.ma.filled(nc.variables[f[1]][id_date, ...], np.nan)
        lg = nc.variables['longitude'][:]
        lt = nc.variables['latitude'][:]
        lons, lats = np.meshgrid(lg, lt)
        nc.close()
        #values = Parallel(n_jobs=cpu)(delayed(extractData)(mat.flatten(), lons.flatten(), lats.flatten(), lcoords, window, reso) for lcoords in list_jobs)
        values = Parallel(n_jobs=cpu)(delayed(extractData_helper)(mat, lg, lt, lcoords, window, reso) for lcoords in list_jobs)
        #matrices.append([mat.flatten(), lons.flatten(), lats.flatten()])
        ##### parallelisation
        #t1 = time.time()
        #values = Parallel(n_jobs=cpu)(delayed(extractData_helper)(matrices, lcoords, window, reso) for lcoords in list_jobs)
        t2 = time.time() - t1
        print t2, ' sec'
        list_values[f[1]] = np.vstack(values)
    #####

    return list_values  

if __name__ == "__main__":

    ddir = "/home/mers/Bureau/teledm/donnees/lidar/calipso/caliop/out"
    os.chdir(ddir)
    files = sorted(glob("*.csv"))
    xo_min, xo_max = -25.0, 57.01
    yo_min, yo_max = -1.25, 51.01
    reso_spatiale = 0.25
    xo = np.arange(xo_min, xo_max, reso_spatiale)  #longitudes du .nc
    yo = np.arange(yo_min, yo_max, reso_spatiale)[::-1]  #latitudes du .nc
    w_interp = 9
    cpu = 3
    d = datetime(2014, 1, 27)
    lidar_params = ["Column_Optical_Depth_Aerosols_532",
                    "Feature_Optical_Depth_532",
                    "Feature_Optical_Depth_Uncertainty_532",
                    "Top_corr",
                    "Base_corr",
                    ]
    methode_ponderation = 'carreDistance'
    stypes = ['dust', 'polluted_dust']
    csv = pd.read_csv(files[0], header=0)

    # aod aerus 0.05 deg
    fnc1 = '/home/mers/Bureau/teledm/donnees/satellite/msg/seviri_aerus/res005/seviri_r005_16d.nc'
    var1 = 'AOD_VIS06'
    # aod MYD04 0.09 deg
    fnc2 = '/home/mers/Bureau/teledm/donnees/satellite/modis/MYD04/res009/MYD04_r009_16d.nc'
    var2 = 'Deep_Blue_Aerosol_Optical_Depth_550_Land'
    # AI omaeruv 0.25 deg
    fnc3 = '/home/mers/Bureau/teledm/donnees/satellite/aura_omi/omaeruv/res025/omaeruv_r025_16d.nc'
    var3 = 'UVAerosolIndex'
    # pDUST chimere02 0.18 deg
    fnc4 = '/home/mers/Bureau/teledm/donnees/modele/wrf/chimere/res018/chimere02_r018_16d_subset.nc'
    var4 = 'mean_pDUST'
    fichiers_ext = [[fnc1,var1], [fnc2,var2], [fnc3, var3], [fnc4,var4]]
    csv_stypes = csv[csv.FeatureSubtype.isin(stypes)].reset_index(drop=True)
    lval = extractData3(methode_ponderation, lidar_params[:], stypes, csv_stypes, fichiers_ext, d, w_interp, cpu, xo, yo, reso_spatiale)

    # export .txt
    
    #exportTXT(lidar_params[:1], fichiers_ext, lval, d, 'dust')
#    ldatas = []
#    for i in range(len(lval[0])):
#        tmp = []
#        for j in range(len(fichiers)):
#            tmp += [fichiers[j][1]]
#            tmp += ['  ']
#            try:
#                tmp += lval[j][i].tolist()
#            except TypeError:
#                tmp.append(lval[j][i])
#            tmp += ['  ///  ']
#        ldatas.append(tmp)
#    with open('/home/mers/Bureau/teledm/donnees/lidar/calipso/caliop/out/info_interp.txt', 'w') as info:
#        [info.write(str(l)+'\n') for l in ldatas]

#    import matplotlib.pyplot as plt
#    from mpldatacursor import datacursor
#    mat = lval[0].reshape(210, -1)
#    datacursor(plt.plot(lidar_lon, lidar_lat, 'k.'))
#    datacursor(plt.imshow(mat, extent=[-25, 57, -0.3, 52], interpolation='none'))
#    plt.colorbar()
#list_values = []
#for p in range(len(lidar_params[:2])):
#    list_values.append(np.asarray([r[i][p][0] for i in range(len(xy))]))
#    list_values.append([r[i][p][1] for i in range(len(xy))])