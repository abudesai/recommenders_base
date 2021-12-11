#!/usr/bin/env python

import paths
from algorithm.train_test_predict import train_model


def train():
    resp = train_model(paths.train_ratings_fpath, paths.model_path, paths.logs_path)
    return resp



if __name__ == '__main__': 
    train() 