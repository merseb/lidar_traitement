#!/bin/sh

cd ${1}
if [[ -f lidar.nc ]]
    then 
    echo le fichier lidar.nc existe deja
    exit
fi
cdo mergetime *.nc lidar.nc
f=$(ls *.csv | head -1)
head -n1 $(ls *.csv | head -1) > tmp1.txt
sed 1d *csv | cat > tmp2.txt
cat tmp1.txt tmp2.txt > lidar.csv
rm tmp*.txt