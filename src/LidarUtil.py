# -*- coding: utf-8 -*-

from pyhdf.SD import SD, SDC
from osgeo import gdal, osr
import numpy as np
import time

####################################################################################
####################################################################################

def splitlist(a, n):
    """
    divise liste en n sous-listes +- egales
    """
    k, m = len(a) / n, len(a) % n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))


####################################################################################
####################################################################################


def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print f.__name__, 'took', end - start, 'time'
        return result
    return f_timer


####################################################################################
####################################################################################




def readHDF(f, lvariables=[]):
    """
    f: fichier hdf
    lvariables(optionnel): liste de variables
    """
    
    hdf = SD(f, SDC.READ)
    if not lvariables:
        lvariables = hdf.datasets().keys()
    for v in lvariables:
        print v
        print hdf.select(v).attributes()
        print hdf.select(v).dimensions(),'\n'
    hdf.end()


####################################################################################
####################################################################################
   

def calcul_Nbpoints(matrice, point):
    """
    Definit le nombre de points lidar dans un intervalle de +- 0.5 degre de latitude
    
    PARAMETRES
    
    **matrice**(*1D array*): ensemble des latitudes \n
    **point**: latitude \n
    
    Retourne le nombre de valeurs trouvees
    """
    latmin =  point - 0.5
    latmax = point + 0.5
    if matrice[0] < matrice[-1]:
        if latmin < matrice[0]:
            latmin = matrice[0]
        if latmax > matrice[-1]:
            latmax = matrice[-1]
    else:
        if latmin < matrice[-1]:
            latmin = matrice[-1]
        if latmax > matrice[0]:
            latmax = matrice[0]
    ind = np.where( (matrice >= latmin) & (matrice <= latmax) )[0]
    return ind.shape[0]

####################################################################################
####################################################################################

#@timefunc
def points2matrice(coord_px,longitude,latitude,valeurs):
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
    idx = np.where((latitude >= y-0.25) & (latitude < y) & (longitude >= x) & (longitude < x+0.25))[0] # recherche des indices des valeurs de lat/lon comprises dans le "pixel" de coord (x[j],y[i])
    if idx.size:
        return np.mean(valeurs[idx])
    else:
        return np.nan



####################################################################################
####################################################################################

    

def indiceCouche(matrice):
    """
    retourne la valeur et l'indice de la 1ere couche ou -9999, -9999 si aucune couche n'est valide

    Parametres:
    **matrice (*2d array*)
    
    """
    ind = np.where(matrice.flatten()[:] != -9999)[0]
    if ind.size:
        return matrice.flatten()[ind[0]],ind[0]
    else:
        return -9999,-9999


####################################################################################
####################################################################################



def array2raster(path,Xo,Yo,pixelWidth,pixelHeight,array):
    """
    Conversion matrice en raster
    
    PARAMETRES:
    
    **path** (*string*): 'path/to/the/file.tif' \n
    **x_origin** (*float*) \n
    **y_origin** (*float*) \n
    **pixelWidth** \n
    **pixelHeight** \n
    **array** : matrice 2d
    
    Retourne fichier .tif
    
    """
    
    cols = array.shape[1]
    rows = array.shape[0]
    reversed_arr = array[::-1] # inversion de la matrice 
    
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(path, cols, rows, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform((Xo, pixelWidth, 0, Yo, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(reversed_arr)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
    
####################################################################################
####################################################################################   

def decodeIGBP(indice):
    IGBPcode = ['Evergreen_Needleleaf_Forest', 'Evergreen_Broadleaf_Forest',
                'Deciduous_Needleleaf_Forest', 'Deciduous_Broadleaf_Forest',
                'Mixed_Forest', 'Closed_Shrublands', 'Open_Shrubland(Desert)',
                'Woody_Savanna', 'Savanna', 'Grassland', 'Wetland', 'Cropland',
                'Urban', 'Crop_Mosaic', 'Permanent_Snow', 'Barren/Desert',
                'Water', 'Tundra']
    return IGBPcode[indice-1]


#if __name__ == "__main__":
    
    ########### test decodeFeatureMask ############################

    
    