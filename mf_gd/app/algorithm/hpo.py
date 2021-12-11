import numpy as np, pandas as pd
import multiprocessing
import time
import sys, os
from random import shuffle
from algorithm.preprocess_pipe import get_preprocess_pipeline
import algorithm.train_test_predict as model_funcs
import algorithm.model_config as cfg

from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
# from skopt.plots import plot_convergence
# from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
from skopt import Optimizer # for the optimization
from joblib import Parallel, delayed # for the parallelization

# HPO
num_CV_folds = 3
num_initial_points = 5
num_searches = 30
max_threads = 10


def get_hp_opt_space(hps): 
    param_grid = []
    for hp_obj in hps: 
        if hp_obj["run_HPO"] == False:
            param_grid.append( Categorical([hp_obj['default']], name=hp_obj['name']) )
        
        elif hp_obj["type"] == 'categorical':
            param_grid.append( Categorical(hp_obj['categorical_vals'], name=hp_obj['name']) )

        elif hp_obj["type"] == 'int' and hp_obj["search_type"] == 'uniform':
            param_grid.append( Integer(hp_obj['range_low'], hp_obj['range_high'], prior='uniform', name=hp_obj['name']) )

        elif hp_obj["type"] == 'int' and hp_obj["search_type"] == 'log-uniform':
            param_grid.append( Integer(hp_obj['range_low'], hp_obj['range_high'], prior='log-uniform', name=hp_obj['name']) )

        elif hp_obj["type"] == 'real' and hp_obj["search_type"] == 'uniform':
            param_grid.append( Real(hp_obj['range_low'], hp_obj['range_high'], prior='uniform', name=hp_obj['name']) )

        elif hp_obj["type"] == 'real' and hp_obj["search_type"] == 'log-uniform':
            param_grid.append( Real(hp_obj['range_low'], hp_obj['range_high'], prior='log-uniform', name=hp_obj['name']) )
        
        else: 
            raise Exception(f"Error creating Hyper-Param Grid. \
                Undefined value type: {hp_obj['type']} or search_type: {hp_obj['search_type']}. \
                Verify hyperparameters.json file.")

    return param_grid  



def get_num_cpus_to_use():
    num_cpus_to_use = min(max_threads, max(multiprocessing.cpu_count() - 2, 1))
    print("num_cpus_to_use: ", num_cpus_to_use)
    return num_cpus_to_use 


def run_hpo(train_ratings_fpath, output_path): 

    start = time.time()

    # multiprocessing doesnt play nice with gpu so disable it. 
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # get training data
    orig_train_data = model_funcs.get_data(train_ratings_fpath)

    # ------------------------------------------------------------------------
    # Get CV folds 
    data_folds = model_funcs.get_cv_fold_data(data = orig_train_data, n_folds = num_CV_folds)
    
    # ------------------------------------------------------------------------
    # get default hyper-parameters
    hps = model_funcs.get_hyper_parameters_json()
    hp_grid = get_hp_opt_space(hps)
    # ------------------------------------------------------------------------
    num_cpus_to_use = get_num_cpus_to_use()
    n_initial_points = max(num_initial_points, num_cpus_to_use)

    optimizer = Optimizer(
        dimensions = hp_grid, # the hyperparameter space
        base_estimator = "GP", # the surrogate
        n_initial_points=n_initial_points, # the number of points to evaluate f(x) to start of
        acq_func='EI', # the acquisition function
        random_state=0, 
        n_jobs=num_cpus_to_use,
    )

    num_loops = int(np.ceil(num_searches / num_cpus_to_use))
    # ------------------------------------------------------------------------
    # define objective
    def objective(sampled_hps_list):      
        hyper_params = collect_sampled_hps(sampled_hps_list, hps)
        losses = []
        for train_data_obj, valid_data_obj in data_folds: 
            model, history, preprocess_pipe = model_funcs.get_trained_model(train_data_obj, hyper_params)

            # for some hyper-param settings, we end up with NaN loss
            last_loss = history.history['loss'][-1]
            if np.isnan(last_loss):
                losses.append(1.0e9)
                break
            
            # pred and score on validation data
            valid_preds_df = model_funcs.predict_with_model(valid_data_obj, model, preprocess_pipe)
            
            loss = model_funcs.get_prediction_score(valid_preds_df)            
            losses.append(loss)

        mean_loss = np.mean(losses)
        print("mean loss = ", mean_loss, hyper_params)
        return mean_loss

    # ------------------------------------------------------------------------
    # x and y in this context mean something different - x refers to hyper-parameter set, and y refers to 
    # resulting validation loss from trained model
    x_dict = {}         
    for i in range(num_loops): 
        print('-'*80)
        x = []; 
        pts = optimizer.ask(n_points=50)
        for pt in pts: 
            pt_key = ''.join(str(s) for s in pt)
            if pt_key in x_dict: continue
            x_dict[pt_key] = 1
            x.append(pt)
            if len(x) == num_cpus_to_use: break

        # print('hp points:', x)
        y = Parallel(n_jobs=num_cpus_to_use)(delayed(objective)(v) for v in x)  # evaluate points in parallel
        optimizer.tell(x, y)
    
    # ------------------------------------------------------------------------
    hpo_results = pd.concat([
        pd.DataFrame(optimizer.Xi),
        pd.Series(optimizer.yi),
    ], axis=1)
    dim_names = collect_hp_names(hps)
    hpo_results.columns = dim_names + [f'val_{cfg.loss_metric}']
    hpo_results.sort_values(by=[f'val_{cfg.loss_metric}'], inplace=True)
    hpo_results.to_csv(os.path.join(output_path, cfg.HPO_RESULTS_FNAME), index=False)
    print(hpo_results.head())
    # ------------------------------------------------------------------------    
    end = time.time()
    print(f"Total HPO time: {np.round((end - start)/60.0, 2)} minutes") 


def collect_sampled_hps(sampled_hps_list, hps):
    hp_dict = {}
    for i, hp_obj in enumerate(hps): 
        hp_dict[hp_obj['name']] = sampled_hps_list[i]
    return hp_dict


def collect_hp_names(hps):
    hp_names = [ hp_obj['name'] for hp_obj in hps]
    return hp_names

if __name__ == "__main__": 
    get_num_cpus_to_use()