# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from glob import glob
from joblib import Parallel, delayed


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


if __name__ == "__main__":
    
    ddir = "/home/mers/Bureau/teledm/donnees/lidar/calipso/caliop/out"
    os.chdir(ddir)
    files = sorted(glob("*filter*.csv"))
    
    # matrice en sortie
    xo = np.arange(-25.,57.01,0.25) 
    yo = np.arange(-1.25,51.01,0.25)[::-1]
    xx,yy = np.meshgrid(xo,yo)
    xy = zip(xx.flatten(),yy.flatten())    
    ###################
    
    
    for f in files[:1]:
        csv = pd.read_csv(f,header=0)
        lon = csv.Longitude.values
        lat = csv.Latitude.values
        lvalues = Parallel(n_jobs=-1)(delayed(points2matrice)(coords,lon,lat,csv.Layer_Base_Altitude) for coords in xy)
        arr = np.asarray(lvalues).reshape(yo.shape[0],-1)
    
    import matplotlib.pyplot as plt
    from mpldatacursor import datacursor
    plt.plot(lon,lat,'k.')
    datacursor(plt.imshow(arr, extent=[xo.min(),xo.max(),yo.min(),yo.max()], interpolation='none'))
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.025)
    plt.minorticks_on()
