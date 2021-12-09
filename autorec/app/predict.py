#!/usr/bin/env python

import paths
from algorithm.train_test_predict import predict_with_model


def predict():
    resp = predict_with_model(paths.test_data_path, paths.model_path, paths.output_path)
    return resp



if __name__ == '__main__': 
    predict() 