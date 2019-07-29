cd /home/t3atlas/ev19u056/projetoWH

module load root
python modelWH1.py -e 100 -a 5 -b 10e-3 -l 8 -v -i 2
git add .
git commit -m "iris_example"
git push
