# -*- coding: utf-8 -*-

from netCDF4 import Dataset, date2index
import pandas as pd
import numpy as np
from datetime import datetime
import os
from glob import glob
import shapefile as shp
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc
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



ddir = os.path.expanduser('~') + '/code/python/lidar_traitement/out'
os.chdir(ddir)
date = '2014-01-27'
w_lissage = 29
couche = 'Base_corr'
couche2 = 'Top_corr'
dbf = '/home/mers/Bureau/teledm/fusion_donnees/shape/dust_pdust_2014_01_27_lissage25v.dbf'
df_shape = dbf2DF(dbf, upper=False)
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


### plot comparaison lissages
couche = ['Base_corr', 'Top_corr', 'Column_Optical_Depth_Aerosols_532']
date = '2014-01-27'
fich = 6
titre = 'Dust, ' + couche[0] + ' ' + couche[1] + ' ' + couche[2] + '  \n' + str(date) + '\nfichier ' + str(fich)
fg, ax = plt.subplots(len(csv), len(couche))
fg.suptitle(titre)
#ax = ax.ravel()
for j in range(len(couche)):
    for i in range(len(csv)):
        df = csv[i][1][(csv[i][1].Date == date) & (csv[i][1].idFile == fich)].reset_index()
        ax[i,j].set_title('lissage ' + csv[i][0] + ' valeurs')
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].plot(df.index, df[couche[j]].values, 'k:', label='init')
        ax[i,j].plot(df.index, df[couche[j] +'_lissage'].values, 'g-', label='init')


plt.savefig(ddir + '/tests/lissages.jpeg', orientation='portrait')



########################################
### plot comparaison couches base / top
titre = 'Dust,  couches ' + couche + '/' + couche2 + '(lissage ' + str(w_lissage) + 'v)'
dates = list(dust.Date.unique())
### plot apport du lissage
f, arr = plt.subplots(8, 2)
f.suptitle(titre)
arr = arr.ravel()
for i in  range(16):
    arr[i].set_title(dates[i])
    arr[i].get_xaxis().set_visible(False)
    arr[i].plot(dust[dust.Date == dates[i]].index, dust[couche][dust.Date == dates[i]].values,'k:', label=couche + 'init')
    arr[i].plot(dust[dust.Date == dates[i]].index, dust[couche + '_lissage'][dust.Date == dates[i]].values, 'r-', label=couche + 'lissage')
    arr[i].plot(dust[dust.Date == dates[i]].index, dust[couche2][dust.Date == dates[i]].values,'k:', label=couche2 + 'init')
    arr[i].plot(dust[dust.Date == dates[i]].index, dust[couche2 + '_lissage'][dust.Date == dates[i]].values, 'g-', label=couche2 + 'lissage')
arr[i].legend(bbox_to_anchor=(1.05, 0), loc='lower center', borderaxespad=0.)



########################################
### plot comparaison 
titre = 'Dust,  couche ' + couche + '(lissages 9,13,21,29 v)'
f, arr = plt.subplots(8, 2)
f.suptitle(titre)
arr = arr.ravel()
for i in  range(16):
    arr[i].set_title(dates[i])
    arr[i].get_xaxis().set_visible(False)
    arr[i].plot(dust[dust.Date == dates[i]].index, dust[couche][dust.Date == dates[i]].values,'k:', label=couche + ' init')
    arr[i].plot(dust[dust.Date == dates[i]].index, dust[couche + '_lissage'][dust.Date == dates[i]].values, 'r-', label=couche + ' (9v)')
    arr[i].plot(dust[dust.Date == dates[i]].index, dust13[couche + '_lissage'][dust.Date == dates[i]].values, 'g-', label=couche + ' (13v)')
    arr[i].plot(dust[dust.Date == dates[i]].index, dust21[couche + '_lissage'][dust.Date == dates[i]].values, 'b-', label=couche + '(21v)')
    arr[i].plot(dust[dust.Date == dates[i]].index, dust29[couche + '_lissage'][dust.Date == dates[i]].values, 'c-', label=couche + '(29v)')
arr[i].legend(bbox_to_anchor=(1.05, 0), loc='lower center', borderaxespad=0.)


i = 0
couche = "Base_corr"
couche2 = "Top_corr"
dusts = [dust, dust13, dust21, dust29, dust35]
lissage = ['9v', '13v', '21v', '29v', '35v']
titre = str(dates[i]) + '\nDust,  couche ' + couche + '(lissages 9,13,21,29,35 v)'
f, arr = plt.subplots(2, 3)
f.suptitle(titre)
arr = arr.ravel()
for j in  range(len(dusts)):
    arr[j].get_xaxis().set_visible(False)
    arr[j].plot(dust[dust.Date == dates[i]].index, dust[couche][dust.Date == dates[i]].values,'k:', label=couche + ' init')
    arr[j].plot(dust[dust.Date == dates[i]].index, dusts[j][couche + '_lissage'][dust.Date == dates[i]].values, 'g-', label=couche + ' (' + lissage[j] + ')')
    arr[j].plot(dust[dust.Date == dates[i]].index, dust[couche2][dust.Date == dates[i]].values,'k--', label=couche2 + ' init')
    arr[j].plot(dust[dust.Date == dates[i]].index, dusts[j][couche2 + '_lissage'][dust.Date == dates[i]].values, 'r-', label=couche2 + ' (' + lissage[j] + ')')
    #arr[j].title(lissage[j])
j=0

for j in range(len(lissage)):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(dust[dust.Date == dates[i]].index, dust[couche][dust.Date == dates[i]].values,'k:', label=couche + ' init')
    ax.plot(dust[dust.Date == dates[i]].index, dusts[j][couche + '_lissage'][dust.Date == dates[i]].values, 'g-', label=couche + ' (' + lissage[j] + ')')
    ax.plot(dust[dust.Date == dates[i]].index, dust[couche2][dust.Date == dates[i]].values,'k--', label=couche2 + ' init')
    ax.plot(dust[dust.Date == dates[i]].index, dusts[j][couche2 + '_lissage'][dust.Date == dates[i]].values, 'r-', label=couche2 + ' (' + lissage[j] + ')')
    ax.legend(), ax.set_title(str(dates[i]) + '\nDust,  couche ' + couche + '/' + couche2 + '(lissage ' + lissage[j] + ' v)')
    fig.savefig(ddir + '/tests/' + couche + '/' + couche2 + '_lissage' + lissage[j] + '.jpeg')

##############################################################################################################################################################
##############################################################################################################################################################
### plot repartition lat/lon dust/polluted_dust
csv = pd.read_csv(files[2], header=0)
latCountDust = []
for l in range(-2, 51):
    latCountDust.append(csv.Latitude[(csv.Latitude.values > l) & (csv.Latitude.values < l+1) & (csv.FeatureSubtype == "dust")].count())

lonCountDust = []
for l in range(-18, 57):
    lonCountDust.append(csv.Longitude[(csv.Longitude.values > l) & (csv.Longitude.values < l+1) & (csv.FeatureSubtype == "dust")].count())

latCountPDust = []
for l in range(-2, 51):
    latCountPDust.append(csv.Latitude[(csv.Latitude.values > l) & (csv.Latitude.values < l+1) & (csv.FeatureSubtype == "polluted_dust")].count())

lonCountPDust = []
for l in range(-18, 57):
    lonCountPDust.append(csv.Longitude[(csv.Longitude.values > l) & (csv.Longitude.values < l+1) & (csv.FeatureSubtype == "polluted_dust")].count())


fig = plt.figure()
#fig.suptitle('Lidar du 2014-01-27 au 2014-02-11')
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Latitude')
ax1.set_ylabel('nb valeurs par deg lat/lon')
ax1.xaxis.label.set_color('red')
ax1.spines['bottom'].set_color('red')
ax1.tick_params(axis='x', colors='red')
#ax2 = ax1.twiny()
#ax2.set_xlabel('Longitude')
#ax2.xaxis.label.set_color('blue')
#ax2.spines['top'].set_color('blue')
#ax2.tick_params(axis='x', colors='blue')
ax1.plot(range(-2,51), latCountDust, 'r', label='Dust')
ax1.plot(range(-2,51), latCountPDust, '--r', label='Polluted Dust')
#ax2.plot(range(-18,57), lonCountDust, 'b', label='dust')
#ax2.plot(range(-18,57), lonCountPDust, '--b', label='Polluted dust')
h1, l1 = ax1.get_legend_handles_labels()
#h2, l2 = ax2.get_legend_handles_labels()
#ax1.legend(h1+h2, l1+l2, loc='upper right')
ax1.legend(h1, l1, loc='upper right')
title = r'\textbf{Lidar du 2014-01-27 au 2014-02-11}'
fig.text(0.5, 1.01, title, horizontalalignment='center', fontsize=12)  #, transform = ax2.transAxes)
plt.show()
fig.savefig(ddir + '/lidar.png')





##############################################################################################################################################################
##############################################################################################################################################################
### plot raster et points pour l'ensemble des orbites

ddirShp = '/home/mers/Bureau/teledm/carto/afrique/'
africa = shp.Reader(ddirShp + 'maskAfricaNorth.shp')
bfa = shp.Reader(ddirShp + 'burkina_faso/BFA_adm0.shp')
mali = shp.Reader(ddirShp + 'mali/MLI_adm0.shp')
benin = shp.Reader(ddirShp + 'benin/BEN_adm0.shp')
niger = shp.Reader(ddirShp + 'niger/NER_adm0.shp')
bbox = [africa, bfa, mali, benin, niger]
ncf = sorted(glob('*lissage' + str(w_lissage) + 'v.nc'))
nc = Dataset(ncf[0], 'r')

dates = nc.variables['time']
ind = date2index(datetime.strptime(date, "%Y_%m_%d"), dates, select='exact')

Pdust = nc.variables['polluted_dust_Concentration_Aerosols'][:]

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




