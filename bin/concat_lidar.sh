#!/bin/sh
suff=${2}_${3}
cd ${1}
if [[ -f lidar_$suff.nc ]]
    then 
    echo le fichier lidar_$suff.nc existe deja
    exit
fi
cdo mergetime *$suff.nc lidar_$suff.nc
f=$(ls *.csv | head -1)
head -n1 $(ls *$suff.csv | head -1) > tmp1.txt
sed 1d *$suff.csv | cat > tmp2.txt
cat tmp1.txt tmp2.txt > lidar_$suff.csv
rm tmp*.txt