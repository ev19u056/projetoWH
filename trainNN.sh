#!/bin/bash
#$ -pe mcore 3

#$ -l container=True
#$ -v CONTAINER=UBUNTU16

#$ -v SGEIN=modelWH.py
#$ -v SGEIN=prepareData.py
#$ -v SGEIN=localConfig.py
#$ -v SGEIN=commonFunctions.py

#$ -v SGEOUT=test:/home/t3atlas/ev19u056/projetoWH

#cd /home/t3atlas/ev19u056/projetoWH

python modelWH.py -e 100 -a 3000 -b 0.0001 -l "64 62 62 62" -v -i 18
python modelWH.py -e 100 -a 3000 -b 0.001 -l "64 62 62 62" -v -i 19
python modelWH.py -e 100 -a 3000 -b 0.01 -l "64 62 62 62" -v -i 20
python modelWH.py -e 100 -a 3000 -b 0.01 -l "64 62 62 62" -v -i 21
python modelWH.py -e 100 -a 3000 -b 0.01 -l "64 62 62 62" -v -i 22
python modelWH.py -e 100 -a 3000 -b 0.01 -l "64 62 62 62" -v -i 23
python modelWH.py -e 100 -a 3000 -b 0.01 -l "64 62 62 62" -v -i 24
python modelWH.py -e 100 -a 3000 -b 0.01 -l "64 62 62 62" -v -i 25
#...$git add .
#...$git commit -m "modelWH.py"
#...$git push
