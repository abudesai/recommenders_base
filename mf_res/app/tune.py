#!/usr/bin/env python

import paths
from algorithm.hpo import run_hpo


def tune():
    resp = run_hpo(paths.train_data_path, paths.output_path)
    return resp



if __name__ == '__main__': 
    tune() 