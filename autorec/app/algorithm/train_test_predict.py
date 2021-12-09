#!/usr/bin/env python
import logging 
import numpy as np, pandas as pd, random
import sys, os, time
import json
import joblib
from sklearn.utils import shuffle
from algorithm.autorec import AutoRec 
from algorithm.preprocess_pipe import get_preprocess_pipeline, get_autorec_pipeline
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
    default_hps = { hp["name"]:hp["default"] for hp in hps if hp["run_HPO"] == True }
    return default_hps


def get_data(data_path): 
    try:
        data = pd.read_csv(data_path)
        return data
    except: 
        raise Exception(f"Error reading data at: {data_path}")


def get_data_based_model_params(R): 
    M = R.shape[1]
    return {"M": M}



def get_trained_model(train_data_tup, valid_data_tup, model_params): 
    print('Training matrix factorizer ...')        
    # Create and train matrix factorization model 
    autorec = AutoRec( **model_params )
    history = autorec.fit(
        train_data_tup = train_data_tup,
        valid_data_tup = valid_data_tup,
        batch_size = 128, 
        epochs = 1,
        verbose = 1, 
    )        
    print('Finished training autorec ...')
    return autorec, history
    


def preprocess_data(data):
    print('Preprocessing train_data ...')
    preprocess_pipe = get_preprocess_pipeline()
    # get ratings and mask matrices
    X, Y, M = preprocess_pipe.fit_transform(data)
    return X, Y, M, preprocess_pipe


def get_train_valid_split(data, valid_split): 
    msk = np.random.rand(len(data)) < valid_split
    valid_data = data[msk].copy()
    train_data = data[~msk].copy()
    return train_data, valid_data


def train_model(train_data_path, model_path, logs_path, random_state=42): 

    print("Starting the training process...")
    start = time.time()

    # set seeds if specified
    set_seeds(seed_value = random_state)        

    # get training data
    orig_train_data = get_data(train_data_path)    
    print('orig_train_data shape:',  orig_train_data.shape)    
    
    # ------------------------------------------------------------------------
    # split train data into train and validation data     
    train_data, valid_data = get_train_valid_split(orig_train_data, cfg.VALIDATION_SPLIT)

    valid_data.sort_values(by=['user_id', 'item_id'], inplace=True)
    print("="*80)
    print(valid_data.head())
    print("="*80)
    print('After train/valid split, train_data shape:',  train_data.shape, 'valid_data shape:',  valid_data.shape)
        
    # ------------------------------------------------------------------------
    # preprocess data
    preprocess_pipe = get_preprocess_pipeline()
    # get ratings and mask matrices
    train_X_R, train_X_M, train_Y_R, train_Y_M = preprocess_pipe.fit_transform(train_data)
    valid_X_R, valid_X_M, valid_Y_R, valid_Y_M = preprocess_pipe.transform(valid_data)
    print('processed train data and mask shape:',  train_X_R.shape, train_X_M.shape, train_Y_R.shape, train_Y_M.shape)
    print('processed valid data and mask shape:',  valid_X_R.shape, valid_X_M.shape, valid_Y_R.shape, valid_Y_M.shape)
    

    # ------------------------------------------------------------------------
    # get M - number of items. It will be used to define the autorec dimension
    data_based_params = get_data_based_model_params(train_X_R)
    print('data based params:',  data_based_params)
    # ------------------------------------------------------------------------
    # get default hyper-parameters
    hps = get_hyper_parameters_json()
    hyper_params = get_default_hps(hps)

    model_params = { **data_based_params, **hyper_params }
    print('model_params:',  model_params)
    # ------------------------------------------------------------------------

    train_data_tup = train_X_R, train_X_M, train_Y_R, train_Y_M
    valid_data_tup = valid_X_R, valid_X_M, valid_Y_R, valid_Y_M

    model, history = get_trained_model( train_data_tup, valid_data_tup, model_params = model_params)    
    
    # ------------------------------------------------------------------------
    # Save the model and processing pipeline     
    # print('Saving model ...')
    # joblib.dump(preprocess_pipe, os.path.join(model_path, cfg.PREPROCESSOR_FNAME))
    # model.save(model_path)    
    # ------------------------------------------------------------------------
    # test valid predictions
    preds = model.predict(valid_X_R, valid_X_M)    
    preds_df = get_inverse_transformation(preds, valid_Y_M, preprocess_pipe)
    # print("preds_df df", preds_df.head())
    
    # print("valid shapes", valid_Y_R.shape, valid_Y_M.shape)

    valid_Y_R = valid_Y_R.toarray()
    act_df = get_inverse_transformation(valid_Y_R, valid_Y_M, preprocess_pipe)
    act_df.rename(columns={"pred_rating_int": "rating_int", "pred_rating": "rating"}, inplace=True)    
    # print("act df", act_df.head())

    print("="*80)
    print("preds_df")
    preds_df['rating_int'] = act_df['rating_int']
    preds_df['rating'] = act_df['rating']
    preds_df.sort_values(by=['user_id', 'item_id'], inplace=True)
    print(preds_df.head())
    print(preds_df.info())

    # print("missing")
    # idx = (preds_df['user_id'] == 0) & (preds_df['item_id'] == 1)
    # print(preds_df.loc[idx].head())

    print('mse', 
        scoring.get_loss( preds_df['rating_int'], preds_df['pred_rating_int'], 'mse' ))
    
    print('\ncorr act pred int', preds_df[['rating_int', 'pred_rating_int']].corr())
    print('\ncorr act pred', preds_df[['rating', 'pred_rating']].corr())

    # ------------------------------------------------------------------------
    print("="*80)
    print("valid")
    print(valid_data.info())
    valid_data = valid_data.merge(preds_df, how='left', on=[cfg.USER_ID_COL, cfg.ITEM_ID_COL])
    print(valid_data.info())

    
    valid_data.sort_values(by=['user_id', 'item_id'], inplace=True)
    print(valid_data.head())
    print(valid_data.columns)
    print(valid_data.shape)
    print('\nvalid corr act pred', valid_data[['rating_x', 'pred_rating']].corr())

    print(valid_data[valid_data.isna().any(axis=1)].head())
    sys.exit()

    sparse_creator = preprocess_pipe[cfg.SPARSE_MATRIX_CREATOR]
    act_df = sparse_creator.inverse_transform(valid_Y_R, valid_Y_M)

    preds_df['rating_int'] = act_df['rating_int']
    print(preds_df.head())

    print('mse', scoring.get_loss( preds_df['rating_int'], preds_df['pred_rating_int'], 'mse' ))
    print('corr int', preds_df[['rating_int', 'pred_rating_int']].corr())
    

    print("-"*80)
    valid_data = valid_data.merge(preds_df, on=[cfg.USER_ID_COL, cfg.ITEM_ID_COL])
    print(valid_data.head())
    print(valid_data.shape)
    print('mse', scoring.get_loss(
        valid_data[cfg.RATING_INT_COL], valid_data[cfg.PRED_RATING_INT_COL], 'mse'
        ))

    print('\ncorr act pred int', valid_data[['rating_int', 'pred_rating_int']].corr())
    print('\ncorr act pred', valid_data[['rating', 'pred_rating']].corr())
    print('\ncorr act act_int', valid_data[['rating', 'rating_int']].corr())
    print('\ncorr pred pred_int', valid_data[['pred_rating', 'pred_rating_int']].corr())
    sys.exit()
    
    # ------------------------------------------------------------------------
    end = time.time()
    print(f"Total training time: {np.round((end - start)/60.0, 2)} minutes") 
    
    sys.exit()
    return 0


def get_inverse_transformation(preds, mask, pipe): 
    sparse_creator = pipe[cfg.SPARSE_MATRIX_CREATOR]
    preds_df = sparse_creator.inverse_transform(preds, mask)

    scaler = pipe[cfg.RATINGS_SCALER]
    preds_df = scaler.inverse_transform(preds_df)

    id_mapper = pipe[cfg.USER_ITEM_MAPPER]
    preds_df = id_mapper.inverse_transform(preds_df)
    return preds_df


def get_prediction_for_batch(model, X, M):
    return model.predict( X )



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

    X = proc_test_data[[cfg.USER_ID_INT_COL, cfg.ITEM_ID_INT_COL]]

    preds = get_prediction_for_batch(mf, X )

    print("preds shape: ", preds.shape)

    scaler = preprocess_pipe[cfg.RATINGS_SCALER]
    proc_test_data[cfg.PRED_RATING_COL] = scaler.inverse_transform(preds)

    final_cols = test_data_cols + [cfg.PRED_RATING_COL]

    proc_test_data = proc_test_data[final_cols]

    proc_test_data.to_csv(os.path.join(output_path, cfg.PREDICTIONS_FNAME), index=False)
    return 0



def load_model_and_preprocessor(model_path):
    if not os.path.exists(os.path.join(model_path, cfg.PREPROCESSOR_FNAME)):
        err_msg = f"No trained model found. Expected to find model files in path: {model_path}"
        print(err_msg)
        return err_msg

    preprocess_pipe = joblib.load(os.path.join(model_path, cfg.PREPROCESSOR_FNAME))
    model = AutoRec.load(model_path)
    return model, preprocess_pipe


def score_predictions(output_path): 
    pred_file = os.path.join(output_path, cfg.PREDICTIONS_FNAME)
    if not os.path.exists(pred_file):
        err_msg = f"No predictions file found. Expected to find: {pred_file}"
        print(err_msg)
        return err_msg

    df = pd.read_csv(pred_file)

    loss_types = ['mse', 'rmse', 'mae', 'nmae', 'smape', 'r2']
    scores_dict = scoring.get_loss_multiple(df[cfg.RATING_COL], df[cfg.PRED_RATING_COL], loss_types)
    print("score", scores_dict)
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

    train_model(train_data_path, model_path, logs_path) 
    


