#!/usr/bin/env python
import logging 
import numpy as np, pandas as pd, random
import sys, os, time
import json
import joblib
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from algorithm.autorec import AutoRec 
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
    default_hps = { hp["name"]:hp["default"] for hp in hps if hp["run_HPO"] == True }
    return default_hps


def get_data(data_path): 
    try:
        return pd.read_csv(data_path)
    except: 
        raise Exception(f"Error reading data at: {data_path}")


def get_data_based_model_params(R): 
    M = R.shape[1]
    return {"M": M}


def get_train_valid_split(data, valid_split): 
    msk = np.random.rand(len(data)) < valid_split
    valid_data = data[msk].copy()
    train_data = data[~msk].copy()
    return train_data, valid_data


def get_cv_fold_data(data, n_folds):
    data_folds = []
    kf = KFold(n_folds)
    for train_index, valid_index in kf.split(data):
        train_data, valid_data = data.iloc[train_index],data.iloc[valid_index]
        data_folds.append(( train_data, valid_data ))
    return data_folds


def get_trained_model(training_data, hyper_params): 

    # ------------------------------------------------------------------------
    # split train data into train and validation data     
    train_data, valid_data = get_train_valid_split(training_data, cfg.VALIDATION_SPLIT)
    # print('After train/valid split, train_data shape:',  train_data.shape, 'valid_data shape:',  valid_data.shape)
        
    # ------------------------------------------------------------------------
    # preprocess data
    print("Pre-processing data...")
    preprocess_pipe = get_preprocess_pipeline()
    # get ratings and mask matrices
    train_X_R, train_X_M, train_Y_R, train_Y_M, train_user_ids_int = preprocess_pipe.fit_transform(train_data)
    valid_X_R, valid_X_M, valid_Y_R, valid_Y_M, valid_user_ids_int = preprocess_pipe.transform(valid_data)
    
    train_data_tup = train_X_R, train_X_M, train_Y_R, train_Y_M
    valid_data_tup = valid_X_R, valid_X_M, valid_Y_R, valid_Y_M

    # ------------------------------------------------------------------------
    # get M - number of items. It will be used to define the autorec dimension
    data_based_params = get_data_based_model_params(train_X_R)
    print('data based params:',  data_based_params)
    
    model_params = { **data_based_params, **hyper_params }
    # print('model_params:',  model_params)
    # ------------------------------------------------------------------------
    print('Training AutoRec ...')        
    # Create and train matrix factorization model 
    autorec = AutoRec( **model_params )
    history = None
    history = autorec.fit(
        train_data_tup = train_data_tup,
        valid_data_tup = valid_data_tup,
        batch_size = 128, 
        epochs = 30,
        verbose = 1, 
    )        
    print('Finished training autorec ...')

    # ------------------------------------------------------------------------
    return autorec, history, preprocess_pipe


def run_training(train_ratings_fpath, model_path, logs_path, random_state=42): 

    print("Starting the training process...")
    start = time.time()

    # set seeds if specified
    set_seeds(seed_value = random_state)        

    # get training data
    orig_train_data = get_data(train_ratings_fpath)    
    print('orig_train_data shape:',  orig_train_data.shape)        
   
    # get default hyper-parameters
    hps = get_hyper_parameters_json()
    hyper_params = get_default_hps(hps)

    # Run training job and get model    
    print(f'Training {cfg.MODEL_NAME} ...')  
    model, train_hist, preprocess_pipe = get_trained_model(orig_train_data, hyper_params)

    # Save the model and processing pipeline     
    print('Saving model ...')
    save_model_and_preprocessor(model, preprocess_pipe, model_path)    
    
    end = time.time()
    print(f"Total training time: {np.round((end - start)/60.0, 2)} minutes") 
    
    return 0


def get_inverse_transformation(preds, mask, users, pipe): 
    sparse_creator = pipe[cfg.SPARSE_MATRIX_CREATOR]
    preds_df = sparse_creator.inverse_transform(preds, mask, users)

    scaler = pipe[cfg.RATINGS_SCALER]
    preds_df = scaler.inverse_transform(preds_df)

    id_mapper = pipe[cfg.USER_ITEM_MAPPER]
    preds_df = id_mapper.inverse_transform(preds_df)
    return preds_df


def predict_with_model(predict_data, model, preprocess_pipe):     
    N = predict_data.shape[0]
    num_batches = (N // cfg.MAX_BATCH_SIZE) if N % cfg.MAX_BATCH_SIZE == 0 else (N // cfg.MAX_BATCH_SIZE) + 1
    # print("num_batches", num_batches); sys.exit()

    all_preds = []
    for i in range(num_batches): 
        
        mini_batch = predict_data.iloc[i*cfg.MAX_BATCH_SIZE : (i+1)*cfg.MAX_BATCH_SIZE, :]

        # transform data
        test_X_R, test_X_M, test_Y_R, test_Y_M, test_users_int_id = preprocess_pipe.transform(mini_batch)
        if test_X_R is None or test_X_R is None: continue

        # print('processed train data and mask shape:',  test_X_R.shape, test_X_M.shape, test_Y_R.shape, test_Y_M.shape)
    
        # make predictions
        preds = model.predict(test_X_R, test_X_M )

        # make inverse transformations on predictions
        preds_df = get_inverse_transformation(preds, test_Y_M, test_users_int_id, preprocess_pipe)

        preds_df = mini_batch.merge(
            preds_df[[cfg.USER_ID_COL, cfg.ITEM_ID_COL, cfg.PRED_RATING_COL]], 
            on=[cfg.USER_ID_COL, cfg.ITEM_ID_COL])   
        
        all_preds.append(preds_df)
    
    if len(all_preds) == 0:
        msg = '''
        Pre-processed prediction data is empty. No predictions to run.
        This usually occurs if none of the users and/or items in prediction data
        were present in the training data. 
        '''
        print(msg)
        return None
    else: 

        all_preds = pd.concat(all_preds, ignore_index=True)
    
    return all_preds


def clear_predictions_dir(output_path):
    for fname in os.listdir(output_path):
        fpath = os.path.join(output_path, fname)
        os.unlink(fpath)



def run_predictions(data_fpath, model_path, output_path):

    # clear previous prediction and score files
    clear_predictions_dir(output_path)

    # get test data
    print("Reading prediction data... ")
    test_data = get_data(data_fpath) 
    print("test data shape: ", test_data.shape)
    

    # load the model, and preprocessor 
    print(f"Loading trained {cfg.MODEL_NAME}... ")
    model, preprocess_pipe = load_model_and_preprocessor(model_path)

    # get predictions from model
    print("Making predictions ...")    
    preds_df = predict_with_model(test_data, model, preprocess_pipe)
        
    print("Saving predictions... ")
    if preds_df is not None:
        preds_df.to_csv(os.path.join(output_path, cfg.PREDICTIONS_FNAME), index=False)
    else: 
        print("No predictions saved.")
    
    print("Done with predictions.")
    return 0


def save_model_and_preprocessor(model, preprocess_pipe, model_path):
    joblib.dump(preprocess_pipe, os.path.join(model_path, cfg.PREPROCESSOR_FNAME))
    model.save(model_path) 
    return    


def load_model_and_preprocessor(model_path):
    if not os.path.exists(os.path.join(model_path, cfg.PREPROCESSOR_FNAME)):
        err_msg = f"No trained model found. Expected to find model files in path: {model_path}"
        print(err_msg)
        return err_msg

    try: 
        preprocess_pipe = joblib.load(os.path.join(model_path, cfg.PREPROCESSOR_FNAME))
    except: 
        raise Exception("Error loading the preprocessor. Do you have the right trained preprocessor?")
    
    try: 
        model = AutoRec.load(model_path)    
    except: 
        raise Exception(f"Error loading the trained {cfg.MODEL_NAME} model. Do you have the right trained preprocessor?")    

    return model, preprocess_pipe


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


    


