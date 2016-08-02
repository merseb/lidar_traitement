# -*- coding: utf-8 -*-

from pyhdf.SD import SD, SDC
from netCDF4 import Dataset, num2date, date2num, date2index
import numpy as np
import pandas as pd
from pyresample import geometry, kd_tree
from glob import glob
import os



ddir = "/home/mers/Bureau/teledm/donnees/lidar/calipso/caliop/zone_etude"
os.chdir(ddir)


x = np.arange(-25.,57.01,0.25)#longitudes du .nc
y = np.arange(-0.3,51.01,0.25)#latitudes du .nc
ext=[min(x),max(x),min(y),max(y)]
gridx, gridy = np.meshgrid(x,y)
grid_def = geometry.GridDefinition(lons=gridx, lats=gridy)# definition de la géométrie de l'image en sortie


series = pd.date_range('2013-12-30','2014-12-31', freq="d").tolist()
listdates = [series[i:i+14] for i in range(0,len(series),14)]
files = {d[0]:[sorted(glob("*"+n.strftime('%Y-%m-%d')+"*.hdf")) for n in d] for d in listdates}
for k in files.keys():
    lf = []
    for l in files[k]:
        lf += l
    files[k] = lf
u_time = 'hours since 1900-01-01 00:00:00.0'
dates = date2num(sorted(files.keys()),u_time)


varlist = ["IGBP_Surface_Type","Day_Night_Flag","DEM_Surface_Elevation","Column_Optical_Depth_Aerosols_532","Feature_Optical_Depth_532","Feature_Optical_Depth_Uncertainty_532","ExtinctionQC_532","CAD_Score","Feature_Classification_Flags","Number_Layers_Found","Layer_Top_Altitude","Layer_Base_Altitude"]
resamplelist = ["DEM_Surface_Elevation","Column_Optical_Depth_Aerosols_532","Feature_Optical_Depth_532","Feature_Optical_Depth_Uncertainty_532","Layer_Top_Altitude","Layer_Base_Altitude"]


   



def indicevalue(matrice):
    ind = np.where(~np.isnan(matrice.flatten()))[0]
    if ind.size:
        return matrice.flatten()[ind[0]]


df = pd.DataFrame(columns=['Latitude','Longitude']+varlist[-1:])
for k in sorted(files.keys())[:1]:
    print "\n\n\n",k
    for f in files[k][:]:
        print "\n",f,"\n"
        hdf = SD(f, SDC.READ)
        lat = hdf.select('Latitude')[:,1]
        lon =hdf.select('Longitude')[:,1]
        idx = np.array(np.where((lat >= min(y)) & (lat <= max(y)) & (lon >= min(x)) & (lon <= max(x))))
        idx = idx.reshape(idx.shape[1])
        if not idx.size:
            print f
            print 'idx vide'
            continue
        else:
            df_tmp = pd.DataFrame()
            df_tmp["Latitude"] = lat[idx]
            df_tmp["Longitude"] = lon[idx]
            for var in varlist[-1:]:
                mat_in = hdf.select(var)
                attribs = mat_in.attributes()
                if hdf.datasets()[var][1][1] == 1:
                    val = mat_in[:].flatten()[idx].astype(float)
                    val[ val == -9999 ] = np.nan
                    df_tmp[var] = val
                else:
                    mat = mat_in[:]
                    val = mat[idx,:].astype(float)
                    val[ val == -9999 ] = np.nan
                    list_mat = np.vsplit(val,val.shape[0])
                    val_tmp = map(indicevalue,list_mat)
                    df_tmp[var] = val_tmp
        mat_in.endaccess()
        hdf.end()
        df_tmp.dropna(inplace=True)
        if not df_tmp.empty:
            df_tmp.to_csv("/home/mers/Bureau/teledm/donnees/lidar/calipso/caliop/tmp/"+f[31:-4]+'.csv',index=False)

