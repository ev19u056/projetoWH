cd /home/t3atlas/ev19u056/projetoWH

module load root-6.10.02
python modelWH1.py -e 100 -a 5 -b 10e-3 -l 8 -v
git add .
git commit -m "iris_example"
git push
