# -*- coding: utf-8 -*-

from netCDF4 import Dataset, date2index
import pandas as pd
import numpy as np
from datetime import datetime
import os, sys
from glob import glob
import shapefile as shp
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc
reload(sys)  
sys.setdefaultencoding('utf8')
#rc('text', usetex=True)
rc('figure', figsize=(11.69,8.27))

import pysal as ps
'''
Arguments
---------
dbfile  : DBF file - Input to be imported
upper   : Condition - If true, make column heads upper case
'''
def dbf2DF(dbfile, upper=True): #Reads in DBF files and returns Pandas DF
    db = ps.open(dbfile) #Pysal to open DBF
    d = {col: db.by_col(col) for col in db.header} #Convert dbf to dictionary
    #pandasDF = pd.DataFrame(db[:]) #Convert to Pandas DF
    pandasDF = pd.DataFrame(d) #Convert to Pandas DF
    if upper == True: #Make columns uppercase if wanted 
        pandasDF.columns = map(str.upper, db.header) 
    db.close() 
    return pandasDF



ddir = os.path.expanduser('~') + '/code/python/lidar_traitement/out/tmp'
os.chdir(ddir)

fdbf = '/home/mers/Bureau/teledm/fusion_donnees/shape/2014_afrique_lissageBase9px.dbf'
dbf = dbf2DF(fdbf, upper=False)
dbf.set_index(pd.to_datetime(dbf.Date), inplace=True)
subT = pd.DataFrame(dbf.FeatureSub.values, index=pd.to_datetime(dbf.Date), columns=['FeatureSubtype'])
nbPdust = dbf.FeatureSub[dbf.FeatureSub=='polluted_dust'].resample('M','count')
nbDust = dbf.FeatureSub[dbf.FeatureSub=='polluted_dust'].resample('M','count')

fig, ax = plt.subplots(6,2, sharex=False, sharey=False, figsize=(23,12)) #, gridspec_kw={'hspace':0, 'wspace':0})
fig.suptitle('Distribution(pourcentage) des donnees lidar sur le continent africain\npar degre de latitude pour chaque mois de 2014')
fig.text(0.1, 0.5, 'Pourcentage valeurs', ha='center', va='center', rotation='vertical')
fig.text(0.5, 0.05, 'Latitude', ha='center', va='center', rotation='horizontal')
for i in range(12):
    if i < 6:
        j = i
        z = 0
    else:
        j = i-6
        z = 1
    nb = dbf.Latitude['2014-'+str(i+1)].count()
    for t in ['dust','polluted_dust']:
        weight = np.ones_like(dbf.Latitude[dbf.FeatureSub==t]['2014-'+str(i+1)])*100/nb
        ax[j,z].hist(dbf.Latitude[dbf.FeatureSub==t]['2014-'+str(i+1)], bins=range(-5,41), weights=weight, label=t, alpha=0.5)
    ax[j,z].text(1, 3., dbf.Latitude['2014-'+str(i+1)].index[0].strftime('%B'), style='italic',bbox={'facecolor':'none', 'alpha':0.5, 'pad':10}, fontsize=10)
    ax[j,z].legend(framealpha=0.5)
plt.show()



files = sorted(glob('*.csv'))
lcsv = []
for f in files[2:3]:
    df = pd.read_csv( f, header=0)
    dust = df[df.FeatureSubtype == "dust"].reset_index()
    pdust = df[df.FeatureSubtype == "polluted_dust"].reset_index()
    lcsv.append([f[-12:-10], dust, pdust])

dates = list(dust.Date.unique())



##############################################################################################################################################################
##############################################################################################################################################################






##############################################################################################################################################################
##############################################################################################################################################################
### plot raster et points pour l'ensemble des orbites
date = '2014-01-27'


ddirShp = '/home/mers/Bureau/teledm/carto/afrique/'
africa = shp.Reader(ddirShp + 'maskAfricaNorth.shp')
bfa = shp.Reader(ddirShp + 'burkina_faso/BFA_adm0.shp')
mali = shp.Reader(ddirShp + 'mali/MLI_adm0.shp')
benin = shp.Reader(ddirShp + 'benin/BEN_adm0.shp')
niger = shp.Reader(ddirShp + 'niger/NER_adm0.shp')
bbox = [africa, bfa, mali, benin, niger]
ncf = sorted(glob('lidar*nc'))
base = []
top = []
aod = []
masse = []
dust = {}
pdust = {}
varLissage = ['Base_corr', 'Top_corr', 'Column_Optical_Depth_Aerosols_532', 'Concentration_Aerosols']
i = 3
for f in ncf:
    nc = Dataset(f, 'r')
    dates = nc.variables['time']
    ind = date2index(datetime.strptime(date, "%Y-%m-%d"), dates, select='exact')
    tpdust = []
    tdust = []
    for i in range(len(varLissage)):
        tpdust.append(nc.variables['polluted_dust_' + varLissage[i]][ind,...])
        tdust.append(nc.variables['dust_' + varLissage[i]][ind,...])
        pdust[varLissage[i]] = tpdust
        dust[varLissage[i]] =tdust
        pdustGrid = nc.variables['polluted_dust_' + varLissage[i]][ind,...]
        dustGrid = nc.variables['dust_' + varLissage[i]][ind,...]
    nc.close()

fig, ax = plt.subplots(2,2, figsize=(23,12))
ax = ax.ravel()
for i in range(4):
    im = ax[i].imshow(dust[i], interpolation='none', vmin=0, vmax=1000)
    #ax.title(varLissage[0])
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()

xo = np.arange(-25.,57.01,0.25) 
yo = np.arange(-1.25,51.01,0.25)[::-1]
xx,yy = np.meshgrid(xo,yo)
xy = zip(xx.flatten(),yy.flatten())  
plt.figure()
for b in bbox:
    for shape in b.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x,y, 'r')
plt.plot(lonPDust,latPDust,'k.')
plt.imshow(Pdust, extent=[xo.min(),xo.max(),yo.min(),yo.max()], interpolation='none')
plt.show(), plt.colorbar()




