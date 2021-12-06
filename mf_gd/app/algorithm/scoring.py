
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import algorithm.model_config as cfg

def get_nmae(Y, Yhat): 
    sum_abs_diff = np.sum(np.abs(Y - Yhat))
    sum_act = Y.sum()
    return sum_abs_diff / sum_act


def get_smape(Y, Yhat):
    return 100./len(Y) * np.sum(2 * np.abs(Yhat - Y) / (np.abs(Y) + np.abs(Yhat)))


def get_wape(Y, Yhat): 
    abs_diff = np.abs(Y - Yhat)
    return 100 * np.sum(abs_diff) / np.sum(Y)


def get_rmse(Y, Yhat):
    return mean_squared_error(Y, Yhat) ** 0.5


def get_loss(Y, Yhat, loss_type = cfg.loss_metric): 
    if loss_type == 'mse':  return mean_squared_error(Y, Yhat)
    if loss_type == 'rmse':  return get_rmse(Y, Yhat)
    elif loss_type == 'mae':  return mean_absolute_error(Y, Yhat)
    elif loss_type == 'nmae':  return get_nmae(Y, Yhat)
    elif loss_type == 'smape':  return get_smape(Y, Yhat)
    elif loss_type == 'r2':  return r2_score(Y, Yhat)
    else: raise Exception(f"undefined loss type: {loss_type}")


loss_funcs = {
    'mse': mean_squared_error,
    'rmse': get_rmse,
    'mae': mean_absolute_error,
    'nmae': get_nmae,
    'smape': get_smape,
    'r2': r2_score,
}


def get_loss_multiple(Y, Yhat, loss_types): 
    scores = {}
    for loss in loss_types:
        scores[loss] = loss_funcs[loss](Y, Yhat)
    return scores