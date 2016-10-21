# -*- coding: utf-8 -*-

from netCDF4 import Dataset, date2index
import pandas as pd
import numpy as np
from datetime import datetime
import os, sys
from glob import glob
import shapefile as shp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
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


ddir_fig ='/home/mers/Bureau/teledm/fusion_donnees/resultats/figures'
ddir = os.path.expanduser('~') + '/code/python/lidar_traitement/out/tmp'
os.chdir(ddir)

fdbf = '/home/mers/Bureau/teledm/fusion_donnees/shape/2014_afrique_lissageBase9px.dbf'
dbf = dbf2DF(fdbf, upper=False)
dbf.set_index(pd.to_datetime(dbf.Date), inplace=True)
subT = pd.DataFrame(dbf.FeatureSub.values, index=pd.to_datetime(dbf.Date), columns=['FeatureSubtype'])
nbPdust = dbf.FeatureSub[dbf.FeatureSub=='polluted_dust'].resample('M').count()
nbDust = dbf.FeatureSub[dbf.FeatureSub=='polluted_dust'].resample('M').count()

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
        weight = np.ones_like(dbf.Latitude[dbf.FeatureSub==t]['2014-'+str(i+1)])/nb
        ax[j,z].hist(dbf.Latitude[dbf.FeatureSub==t]['2014-'+str(i+1)], bins=range(-5,41), label=t, alpha=0.5)
    ax[j,z].text(1, 3., dbf.Latitude['2014-'+str(i+1)].index[0].strftime('%B'), style='italic',bbox={'facecolor':'none', 'alpha':0.5, 'pad':10}, fontsize=10)
    ax[j,z].legend(framealpha=0.5)
plt.show()

date = '2014-01'
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(23,12))
ax1.hist(dbf.Latitude[date], bins=range(-5,41), label='dust + polluted dust', alpha=0.5)
ax1.hist(dbf.Latitude[dbf.FeatureSub=='dust'][date], bins=range(-5,41), histtype='bar', label='dust', alpha=0.5)
ax1.hist([dbf.Latitude[dbf.FeatureSub=='polluted_dust'][date]],bins=range(-5,41), histtype='bar', stacked=True, label='polluted dust', alpha=0.5)

ax2.hist([dbf.Latitude[dbf.FeatureSub=='dust'][date],dbf.Latitude[dbf.FeatureSub=='polluted_dust'][date]],bins=range(-5,41), histtype='bar', stacked=True, label=['dust','polluted dust'], alpha=0.5)
w = np.ones_like(dbf.Latitude[date])*100/(nbDust[0]+nbPdust[0])
wDust = np.ones_like(dbf.Latitude[dbf.FeatureSub=='dust'][date])*100/nbDust[0]
wPdust = np.ones_like(dbf.Latitude[dbf.FeatureSub=='polluted_dust'][date])*100/nbPdust[0]
ax3.hist(dbf.Latitude[date], bins=range(-5,41), weights=w, label='dust + polluted dust', alpha=0.5)
ax3.hist(dbf.Latitude[dbf.FeatureSub=='dust'][date], bins=range(-5,41), weights=wDust, label='dust', alpha=0.5)
ax3.hist([dbf.Latitude[dbf.FeatureSub=='polluted_dust'][date]],bins=range(-5,41), weights=wPdust, label='polluted dust', alpha=0.5)

ax1.legend()
ax2.legend()
fig.show()
plt.show()


dbf.Latitude[date].hist(bins=range(-5,41), label='dust + polluted dust', alpha=0.5)
dbf.Latitude[dbf.FeatureSub=='dust'][date].hist(bins=range(-5,41), label='dust', alpha=0.5)
dbf.Latitude[dbf.FeatureSub=='polluted_dust'][date].hist(bins=range(-5,41), label='polluted dust', alpha=0.5)
plt.legend()
plt.show()
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



varLissage = ['Base_corr', 'Top_corr', 'Column_Optical_Depth_Aerosols_532', 'Concentration_Aerosols']

varL = 'Concentration_Aerosols'
dust, pdust = [], []
f = [n for n in ncf if varL in n][0]
nc = Dataset(f, 'r')
dates = nc.variables['time']
ind = date2index(datetime.strptime(date, "%Y-%m-%d"), dates, select='exact')
pdustGrid = nc.variables['polluted_dust_Base_corr_point2grid'][ind,...]
dustGrid = nc.variables['dust_Base_corr_point2grid'][ind,...]
for v in varLissage:
    dust.append(nc.variables['dust_' + v][ind,...])
    pdust.append(nc.variables['polluted_dust_' + v][ind,...])
nc.close()


axs = ['ax'+str(i) for i in range(4)]
ims = ['im'+str(i) for i in range(4)]
caxs = ['cax'+str(i) for i in range(4)]
minmax = [[0,4], [0,7], [0,1], [0,1000]]
for p in [[dust, 'Dust'], [pdust,'PollutedDust']]:
    fig = plt.figure(1, figsize=(23,12))
    fig.suptitle(p[1] + '\n' + date + ' Interpolation apres lissage de la variable ' + varL)
    for idx in range(len(axs)):
        axs[idx] = fig.add_subplot(2, 2, (idx + 1))
        divider = make_axes_locatable(axs[idx])
        caxs[idx] = divider.append_axes("right", size = "5%", pad = 0.05)
        ims[idx] = axs[idx].imshow(p[0][idx], vmin=minmax[idx][0], vmax=minmax[idx][1], interpolation='none')
        axs[idx].set_title(varLissage[idx]+'\n(min '+ str(p[0][idx].min()) + ' max ' + str(p[0][idx].max()) + ')')
        plt.colorbar(ims[idx], cax = caxs[idx])
    plt.show()
    fig.savefig(ddir_fig + '/'+ date.replace('-', '') + '_interpolations' + p[1] + '_lissage' + varL + '.png', dpi=330)
    plt.clf()


xo = np.arange(-25.,57.01,0.25) 
yo = np.arange(-1.25,51.01,0.25)[::-1]
xx,yy = np.meshgrid(xo,yo)
xy = zip(xx.flatten(),yy.flatten())  

for stype in ['dust', 'polluted_dust']:
    lon = dbf.Longitude[(dbf.FeatureSub==stype) & (dbf.idPeriod==date)].values
    lat = dbf.Latitude[(dbf.FeatureSub==stype) & (dbf.idPeriod==date)].values
    nc = Dataset(f, 'r')
    dates = nc.variables['time']
    ind = date2index(datetime.strptime(date, "%Y-%m-%d"), dates, select='exact')
    grid = nc.variables[stype + '_Base_corr_point2grid'][ind,...]
    nc.close()
    plt.clf()
    fig = plt.figure(figsize=(23,12))
    fig.suptitle(date + ' ' + stype)
    for b in bbox:
        for shape in b.shapeRecords():
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            plt.plot(x,y, 'r')
    plt.plot(lon,lat,'k.')
    plt.imshow(grid, extent=[xo.min(),xo.max(),yo.min(),yo.max()], interpolation='none')
    plt.show(), plt.colorbar()
    fig.savefig(ddir_fig + '/'+ date.replace('-', '') + '_reechantillonnage_' + stype + '.png', dpi=330)



############## plot interpolations
date='2014-01-27'
files = sorted(glob('*nc'))
types = ['dust', 'polluted_dust']
titles=['(sans lissage)', '(lissage Base)', '(lissage ColAOD)', '(lissage Concentration)', '(lissage Top)']
fig = plt.figure(1,figsize=(23,12))
gs = gridspec.GridSpec(2,5,hspace=0) #width_ratios=[20,10], height_ratios=[10,5])
fig.suptitle(date)
for i in range(5):
    for j in range(2):
        nc = Dataset(files[i], 'r')
        dates = nc.variables['time']
        lon = nc.variables['longitude'][:]
        lat = nc.variables['latitude'][:]
        ind = date2index(datetime.strptime(date, "%Y-%m-%d"), dates, select='exact')
        v = nc.variables[types[j]+'_Concentration_Aerosols'][ind,:,:]
        nc.close()
        ax = plt.subplot(gs[j,i])
        im = ax.imshow(v, extent=[lon.min(), lon.max(), lat.min(), lat.max()], vmin=0, vmax=1500, interpolation='none')
        if i == 4 and j==0:
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, orientation='horizontal')
        #plt.clim(im, vmin=0, vmax=1500)
        ax.set_title(types[j] + 'Concentration \n'+titles[i])
fig.show()
plt.show()
