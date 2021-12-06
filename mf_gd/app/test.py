import algorithm.hpo as hpo



import os

fpath = './../../ml_vol/data/train/ratings_train.csv'
out_path = './'
hpo.run_hpo(fpath, out_path)