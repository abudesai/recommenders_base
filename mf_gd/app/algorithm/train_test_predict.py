#!/usr/bin/env python
import logging 
import numpy as np, pandas as pd, random
import sys, os, time
import json
import joblib
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from algorithm.matrix_fact import MatrixFactorizer
from algorithm.preprocess_pipe import get_preprocess_pipeline
import algorithm.model_config as cfg
import algorithm.scoring as scoring
import tensorflow as tf

hp_f_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.json')


def set_seeds(seed_value=42):
    if type(seed_value) == int or type(seed_value) == float:          
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)


def get_hyper_parameters_json():
    try:
        hps = json.load(open(hp_f_path))        
    except: 
        raise Exception(f"Error reading hyperparameters file at: {hp_f_path}")   
    return hps


def get_default_hps(hps):
    default_hps = { hp["name"]:hp["default"] for hp in hps }
    return default_hps


def get_data(data_path): 
    try:
        return pd.read_csv(data_path)
    except: 
        raise Exception(f"Error reading data at: {data_path}")


def get_data_based_model_params(data): 
    N = int(data.iloc[:, 0].max()+1)
    M = int(data.iloc[:, 1].max()+1)
    return {"N":N, "M": M}


def preprocess_data(data):
    print('Preprocessing train_data ...')
    preprocess_pipe = get_preprocess_pipeline()
    train_data = preprocess_pipe.fit_transform(data)
    X = train_data[[cfg.USER_ID_INT_COL, cfg.ITEM_ID_INT_COL]]
    y = train_data[[cfg.RATING_INT_COL]]
    return X, y, preprocess_pipe


def get_cv_fold_data(data, n_folds):
    data_folds = []
    kf = KFold(n_folds)
    for train_index, valid_index in kf.split(data):
        train_data, valid_data = data.iloc[train_index], data.iloc[valid_index]
        data_folds.append(( train_data, valid_data ))
    return data_folds


def get_trained_model(training_data, hyper_params):     
    # preprocess data
    print("Pre-processing data...")
    X, y, preprocess_pipe = preprocess_data(training_data)
    
    # get model parameters 
    data_based_params = get_data_based_model_params(X)
    model_params = { **data_based_params, **hyper_params }
          
    # Create and train matrix factorization model     
    print('Training model ...')        
    model = MatrixFactorizer( **model_params )
    history = model.fit(
        X = X,  y = y,
        validation_split=cfg.VALIDATION_SPLIT,
        batch_size = 128, 
        epochs = 1,
        verbose = 1, 
    )     
    print('Finished training autorec ...')   
    return model, history, preprocess_pipe


def run_training(train_data_path, model_path, logs_path, random_state=42): 

    print("Starting the training process...")
    start = time.time()

    # set seeds if specified
    set_seeds(seed_value = random_state)        

    # get training data
    orig_train_data = get_data(train_data_path)    
    print('orig_train_data shape:',  orig_train_data.shape)    
    
    # get default hyper-parameters
    hps = get_hyper_parameters_json()
    hyper_params = get_default_hps(hps)
    
    print(f'Training {cfg.MODEL_NAME} ...')  
    model, train_hist, preprocess_pipe = get_trained_model(orig_train_data, hyper_params)

    # Save the model and processing pipeline     
    print('Saving model ...')
    save_model_and_preprocessor(model, preprocess_pipe, model_path)    
    
    end = time.time()
    print(f"Total training time: {np.round((end - start)/60.0, 2)} minutes") 
    
    return 0


def predict_with_model(predict_data, model, preprocess_pipe):
   
    test_data_cols = list(predict_data.columns)    

    # transform data
    proc_test_data = preprocess_pipe.transform(predict_data)
    print("proc_test_data shape: ", proc_test_data.shape)

    X = proc_test_data[[cfg.USER_ID_INT_COL, cfg.ITEM_ID_INT_COL]]

    preds = model.predict( X )

    scaler = preprocess_pipe[cfg.RATINGS_SCALER]
    proc_test_data[cfg.PRED_RATING_COL] = scaler.inverse_transform(preds)

    final_cols = test_data_cols + [cfg.PRED_RATING_COL]

    proc_test_data = proc_test_data[final_cols]
    print("preds shape: ", preds.shape)
    
    return proc_test_data


def run_predictions(data_fpath, model_path, output_path):
     # get data
    print("Reading prediction data... ")
    test_data = get_data(data_fpath) 
    print("test data shape: ", test_data.shape)

    # load the model, and preprocessor 
    print(f"Loading trained {cfg.MODEL_NAME}... ")
    model, preprocess_pipe = load_model_and_preprocessor(model_path)

    # get predictions from model
    print("Making predictions... ")
    preds_df = predict_with_model(test_data, model, preprocess_pipe)
    
    print("Saving predictions... ")
    preds_df.to_csv(os.path.join(output_path, cfg.PREDICTIONS_FNAME), index=False)

    print("Done with predictions.")
    return 0

    
def save_model_and_preprocessor(model, preprocess_pipe, model_path):
    joblib.dump(preprocess_pipe, os.path.join(model_path, cfg.PREPROCESSOR_FNAME))
    model.save(model_path) 
    return    
    

def load_model_and_preprocessor(model_path):
    if not os.path.exists(os.path.join(model_path, cfg.PREPROCESSOR_FNAME)):
        err_msg = f"No trained preprocessor found. Expected to find model files in path: {model_path}"
        print(err_msg)
        return err_msg

    try: 
        preprocess_pipe = joblib.load(os.path.join(model_path, cfg.PREPROCESSOR_FNAME))
    except: 
        raise Exception("Error loading the preprocessor. Do you have the right trained preprocessor?")
    
    try: 
        mf = MatrixFactorizer.load(model_path)        
    except: 
        raise Exception(f"Error loading the trained {cfg.MODEL_NAME} model. Do you have the right trained preprocessor?")
    
    return mf, preprocess_pipe


def get_prediction_score(preds_df): 
    loss = scoring.get_loss(preds_df[cfg.RATING_COL], preds_df[cfg.PRED_RATING_COL], cfg.loss_metric)
    return loss


def score_predictions(output_path): 
    pred_file = os.path.join(output_path, cfg.PREDICTIONS_FNAME)
    if not os.path.exists(pred_file):
        err_msg = f"No predictions file found. Expected to find: {pred_file}"
        print(err_msg)
        return err_msg

    df = pd.read_csv(pred_file)

    cols = df.columns

    if cfg.PRED_RATING_COL not in cols:
        err_msg = f"Prediction file missing column '{cfg.PRED_RATING_COL}'. Cannot generate scores."
        print(err_msg)
        return err_msg
    elif cfg.RATING_COL not in cols:
        err_msg = f"Prediction file missing column '{cfg.RATING_COL}'. Cannot generate scores."
        print(err_msg)
        return err_msg
    else: 
        loss_types = ['mse', 'rmse', 'mae', 'nmae', 'smape', 'r2']
        scores_dict = scoring.get_loss_multiple(df[cfg.RATING_COL], df[cfg.PRED_RATING_COL], loss_types)
        print("scores:", scores_dict)
        with open(os.path.join(output_path, cfg.SCORING_FNAME), 'w') as f: 
            f.write("Attribute,Value\n")
            f.write(f"Model_Name,{cfg.MODEL_NAME}\n")
            for loss in loss_types:
                f.write( f"{loss},{round(scores_dict[loss], 4)}\n" )

    return 0


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

    run_training(train_data_path, model_path, logs_path) 
    


