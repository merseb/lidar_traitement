# -*- coding: utf-8 -*-


import numpy as np
from netCDF4 import Dataset, date2index
from joblib import Parallel, delayed, load, dump, Memory
import time
import os
from glob import glob
import shutil
import tempfile


def splitlist(a, n):
    """
    retourne un iterateur pour diviser une liste a en n sous-listes +- egales
    """
    k, m = len(a) / n, len(a) % n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))


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


def distance(xa, ya, xb, yb):
    """
    """
    return np.sqrt((xb-xa)**2 + (yb-ya)**2)
    

def calcPonderation(weight, valeurs):
    """
    """
    vmoy = np.average(valeurs, weights=weight)
    vmin = np.min(valeurs * weight)
    vmax = np.max(valeurs * weight)
    variance = np.average((valeurs-vmoy)**2, weights=weight)
    vstd = np.sqrt(variance)
    return [np.asarray([valeurs.shape[0], vmoy, vmin, vmax, vstd]), valeurs]


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


def extractData_helper(matrice, longitude, latitude, coords, window, reso):
    """
    Fonction qui permet d'envoyer la fonction rolling_window sous forme de list comprehension
    """
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
        return stats

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
    tmpdir = os.getcwd()

    ##### liste de coordonnees
    xx, yy = np.meshgrid(x, y) # produit cartesien des lon/lat
    xy = zip(xx.flatten(), yy.flatten()) # liste de tuples(lon/lat)

    ##### repartition des couples lat/lon dans n sous-listes = nombre de processeurs pour la parallelisation
    list_jobs = [job for job in splitlist(xy, cpu)] 

    print 'lidar'
    
    ##### boucle sur la liste des sous-types 
    list_values = {} # initialisation du dictionnaire
    subtypes = list(lidar_df.FeatureSubtype.unique())  # liste des sous-categories
    list_df = [ lidar_df[lidar_df.FeatureSubtype == st].reset_index(drop=True) for st in subtypes]  # liste des df par sous-categorie
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
            mat = np.ma.filled(nc.variables[f[1]][id_date,0, ...], np.nan)  # variable 4d: t,z,y,x
        else:
            mat = np.ma.filled(nc.variables[f[1]][id_date, ...], np.nan)  # variable 3d: t,y,x
        filename_mat = os.path.join(tempfile.mkdtemp(prefix='temporaire_', dir=tmpdir), 'newfile.dat')
        m = np.memmap(filename_mat, dtype= mat.dtype, mode='w+', shape=mat.shape)
        m[:] = mat[:]
        lg = nc.variables['longitude'][:]
        filename_lg = os.path.join(tempfile.mkdtemp(prefix='temporaire_', dir=tmpdir), 'newfile.dat')
        lons = np.memmap(filename_lg, dtype= mat.dtype, mode='w+', shape=lg.shape)
        lons[:] = lg[:]
        lt = nc.variables['latitude'][:]
        filename_lt= os.path.join(tempfile.mkdtemp(prefix='temporaire_', dir=tmpdir), 'newfile.dat')
        lats = np.memmap(filename_lt, dtype= mat.dtype, mode='w+', shape=lt.shape)
        lats[:] = lt[:]
        nc.close()
        ##### parallelisation
        values = Parallel(n_jobs=cpu)(delayed(extractData_helper)(m, lons, lats, lcoords, window, reso) for lcoords in list_jobs)
        t2 = time.time() - t1
        print t2, ' sec'
        list_values[f[1]] = np.vstack(values)  # empilement des matrices en sortie
    dir_list = glob.glob(os.path.join(tmpdir, "temporaire_*"))
    for path in dir_list:
        if os.path.isdir(path):
            shutil.rmtree(path)
    #####
    return list_values
