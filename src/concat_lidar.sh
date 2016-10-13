#!/bin/sh

#2 = year, 3 = methode de ponderation, 4 = nb pixel, 5 = periode, 6 = fenetre de lissage
suff=${3}_${4}_${5}d_lissage${6}v
suff1=${2}_${3}_${4}_${5}d_lissage${6}v
#cd ${1}
if [[ -f lidar_$suff.nc ]]
    then 
    echo le fichier lidar_$suff1.nc existe deja
    exit
fi
#cdo mergetime *$suff.nc lidar_$suff1.nc
#rm 2*.nc
f=$(ls *.csv | head -1)
head -n1 $(ls 2*.csv | head -1) > tmp1.txt
sed 1d 2*.csv | cat > tmp2.txt
cat tmp1.txt tmp2.txt > lidar_${2}_${5}d.csv
rm tmp*.txt