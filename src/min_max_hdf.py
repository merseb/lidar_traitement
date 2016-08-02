# -*- coding: utf-8 -*-

import pandas as pa
import numpy as np
import time, datetime
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset, date2num, num2date
import os, shutil, glob, subprocess, sys
from pyresample import kd_tree, geometry
import rasterio
import matplotlib.pyplot as plt


prd = "MYD07"
site = "nasa"
ddir = "/archive/crct/shared/AEROSOL/donnees/satellite/modis/"+prd+"/2007"
os.chdir(ddir)
files = sorted(glob.glob("*.hdf"))
#for f in files[:200]:
#    print f
if prd == "MYD04":
    lv3d = ['Deep_Blue_Aerosol_Optical_Depth_550_Land','Deep_Blue_Angstrom_Exponent_Land','AOD_550_Dark_Target_Deep_Blue_Combined','Deep_Blue_Aerosol_Optical_Depth_550_Land_Best_Estimate','Deep_Blue_Aerosol_Optical_Depth_550_Land_STD','Deep_Blue_Algorithm_Flag_Land','Deep_Blue_Aerosol_Optical_Depth_550_Land_QA_Flag','Deep_Blue_Aerosol_Optical_Depth_550_Land_Estimated_Uncertainty','Deep_Blue_Cloud_Fraction_Land','Deep_Blue_Number_Pixels_Used_550_Land']
    lv4d = ['Deep_Blue_Spectral_TOA_Reflectance_Land','Deep_Blue_Spectral_Surface_Reflectance_Land','Deep_Blue_Spectral_Single_Scattering_Albedo_Land']
    lv = lv3d+lv4d
elif prd == "MYD07":
    lv = ['Total_Ozone','Lifted_Index','Skin_Temperature']
else:
    lv = ['Water_Vapor_Infrared']

for varname in lv:
    ftxt = open("/work/crct/se5780me/"+site+"_"+prd+"_"+varname+"_info.txt","w")
    for f in files[:100]:
        print f
        print varname
        ftxt.write(f+"\n")
        ftxt.write(varname+"\n")
        hdf = SD(f, SDC.READ)
        vari = hdf.select(varname)
        attrib = vari.attributes()
        ftxt.write(str(attrib)+"\n")
        v_mask = np.ma.masked_values(vari[:],attrib['_FillValue'])
        ftxt.write("\n     valeur min avant traitement = "+str(np.nanmin(v_mask))+"\n")
        ftxt.write("     valeur max avant traitement = "+str(np.nanmax(v_mask))+"\n")
        	
        v = vari[:].astype('float')
        v[v==float(attrib['_FillValue'])] = np.nan
        v[:] = (v[:]-attrib['add_offset'])*attrib['scale_factor']
        
        ftxt.write("     valeur min apres traitement = "+str(np.nanmin(v))+"\n")
        ftxt.write("     valeur max apres traitement = "+str(np.nanmax(v))+"\n\n\n")
        	
    hdf.end()
    
    ftxt.close()

