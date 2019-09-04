!/bin/bash
$ -pe mcore 3

$ -l container=True
$ -v CONTAINER=UBUNTU16

$ -v SGEIN=modelWH.py
$ -v SGEIN=prepareData.py
$ -v SGEIN=localConfig.py
$ -v SGEIN=commonFunctions.py

$ -v SGEOUT=test:/home/t3atlas/ev19u056/projetoWH

cd /home/t3atlas/ev19u056/projetoWH
python plotNN.py -v -f Model_Ver_16 -a
python plotNN.py -v -f Model_Ver_17 -a
$git add .
$git commit -m "modelWH.py"
$git push
python plotNN.py -v -f Model_Ver_18 -a
python plotNN.py -v -f Model_Ver_19 -a
$git add .
$git commit -m "modelWH.py"
$git push
python plotNN.py -v -f Model_Ver_20 -a
python plotNN.py -v -f Model_Ver_21 -a
python plotNN.py -v -f Model_Ver_22 -a
$git add .
$git commit -m "modelWH.py"
$git push
python plotNN.py -v -f Model_Ver_23 -a
python plotNN.py -v -f Model_Ver_24 -a
python plotNN.py -v -f Model_Ver_25 -a
$git add .
$git commit -m "modelWH.py"
$git push
python plotNN.py -v -f Model_Ver_26 -a
python plotNN.py -v -f Model_Ver_27 -a
python plotNN.py -v -f Model_Ver_28 -a
$git add .
$git commit -m "modelWH.py"
$git push
#...$git add .
#...$git commit -m "modelWH.py"
#...$git push
