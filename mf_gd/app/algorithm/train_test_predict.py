#!/usr/bin/env python
import logging 
import numpy as np, pandas as pd, random
import sys, os, time
import json
import joblib
from sklearn.utils import shuffle
from algorithm.matrix_fact import MatrixFactorizer
from algorithm.preprocess_pipe import get_preprocess_pipeline
import algorithm.model_config as cfg
import tensorflow as tf

hp_f_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.json')


def set_logging(logs_path):
    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(logs_path, 'tensorflow.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def set_seeds(seed_value=42):   
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)


def get_hyper_parameters():
    try:
        hps = json.load(open(hp_f_path))        
    except: 
        raise Exception(f"Error reading hyperparameters file at: {hp_f_path}")    
    hp_dict = { hp["name"]:hp["default"] for hp in hps }
    return hp_dict


def get_data(data_path): 
    try:
        data = pd.read_csv(data_path)
        return data
    except: 
        raise Exception(f"Error reading data at: {data_path}")
    


def train_model(train_data_path, model_path, logs_path, random_state=42): 

    print("Starting the training process...")

    # set seeds if specified
    if type(random_state) == int or type(random_state) == float:
        set_seeds(seed_value = random_state)

    # get default hyper-parameters
    hp_dict = get_hyper_parameters()

    # get training data
    orig_train_data = get_data(train_data_path)

    # set logging
    set_logging(logs_path)
    
    start = time.time()
    print('train_data shape:',  orig_train_data.shape)
    # ------------------------------------------------------------------------
    # preprocess data
    print('Preprocessing train_data ...')
    preprocess_pipe = get_preprocess_pipeline()
    train_data = preprocess_pipe.fit_transform(orig_train_data)
    print('processed train_data shape:',  train_data.shape)

    N = int(train_data[cfg.USER_ID_INT_COL].max()+1)
    M = int(train_data[cfg.ITEM_ID_INT_COL].max()+1)
    print(f"Found # Users N = {N}; # Items M = {M} in training data")

    # ------------------------------------------------------------------------
    # split train data into train and validation data 
    print('Doing train and validation split ...')
    valid_split = cfg.VALIDATION_SPLIT
    train_data = shuffle(train_data)

    cutoff = int(valid_split*len(train_data))
    valid_data = train_data.iloc[:cutoff]
    train_data = train_data.iloc[cutoff:]
   
    # ------------------------------------------------------------------------
    # Create matrix factorization model     
    print('Instantiating matrix factorizer ...')

    mf = MatrixFactorizer( N=N, M=M, **hp_dict )
    # mf.summary(); sys.exit()

    # ------------------------------------------------------------------------
    # Fit the model to training data 
    print('Fitting matrix factorizer ...')

    _ = mf.fit(
        user_ids = train_data[cfg.USER_ID_INT_COL],
        item_ids = train_data[cfg.ITEM_ID_INT_COL],
        ratings = train_data[cfg.RATING_INT_COL],
        validation_data=(
            [ valid_data[cfg.USER_ID_INT_COL], valid_data[cfg.ITEM_ID_INT_COL] ],
            valid_data[cfg.RATING_INT_COL]
        ),
        batch_size = 128, 
        epochs = 30,
        verbose = 1, 
    )

    # ------------------------------------------------------------------------
    # Save the model and processing pipeline     
    print('Saving model ...')
    joblib.dump(preprocess_pipe, os.path.join(model_path, cfg.PREPROCESSOR_FNAME))
    mf.save(model_path)    
    # ------------------------------------------------------------------------
    end = time.time()
    print(f"Total training time: {np.round((end - start)/60.0, 2)} minutes") 
    
    return 0




def predict_with_model(test_data_path, model_path, output_path):
    # get test data
    test_data = get_data(test_data_path)
    print("test data shape: ", test_data.shape)

    test_data_cols = list(test_data.columns)

    # load the model, and preprocessor 
    mf, preprocess_pipe = load_model_and_preprocessor(model_path)

    # transform data
    proc_test_data = preprocess_pipe.transform(test_data)
    print("proc_test_data shape: ", proc_test_data.shape)

    preds = mf.predict(
        user_ids = proc_test_data[cfg.USER_ID_INT_COL],
        item_ids = proc_test_data[cfg.ITEM_ID_INT_COL],)

    print("preds shape: ", preds.shape)

    scaler = preprocess_pipe[cfg.RATINGS_SCALER]
    proc_test_data[cfg.PRED_RATING_COL] = scaler.inverse_transform(preds)

    final_cols = test_data_cols + [cfg.PRED_RATING_COL]

    proc_test_data = proc_test_data[final_cols]

    proc_test_data.to_csv(os.path.join(output_path, cfg.PREDICTIONS_FNAME), index=False)
    return 0




def load_model_and_preprocessor(model_path):
    preprocess_pipe = joblib.load(os.path.join(model_path, cfg.PREPROCESSOR_FNAME))
    mf = MatrixFactorizer.load(model_path)
    return mf, preprocess_pipe


if __name__ == '__main__': 

    # # dataset = 'movielens-20m'      # jester, movielens-10m
    # # dataset = 'movielens-10m'      # jester, movielens-10m
    # dataset = 'jester'      # jester, movielens-10m

    # data_type = 'train'
    # train_data = pd.read_csv(f'./../../data/{dataset}/processed/{data_type}/ratings_{data_type}.csv')
    # print(train_data.shape)

    # data_type = 'test'
    # test_data = pd.read_csv(f'./../../data/{dataset}/processed/{data_type}/ratings_{data_type}.csv')
    # print(test_data.shape)

    

    train_data_path = '.'
    model_path = './delete/'
    logs_path = './delete/'

    train_model(train_data_path, model_path, logs_path)
    


