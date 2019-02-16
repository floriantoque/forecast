"""
        SARIMAX is a statistical model for time series forecasting..
        Input: Observation per date of each time-series, exogenous data
"""

import numpy as np
import pandas as pd
import sys 
sys.path.insert(0, '../../utils/')
import utils
import utils_date

import itertools
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error
from math import sqrt

#import matplotlib.pyplot as plt

from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing

#import seaborn
#seaborn.set_style('whitegrid')

from tqdm import tqdm
import ast


def create_Xy_basic(start, end, obs, fea, index='Datetime'):
    """
            Extract values from dataframe obs to y variable 
            and values of dataframe fea to X variable
    """
    
    date_list = utils_date.get_list_common_date(start, end, obs, [fea])
    
    y = obs.set_index(index).loc[date_list].values
    X = fea.set_index(index).loc[date_list].values
    mask = (X!=0).sum(axis=0)!=0
    
    X = X[:,mask]
    y_name_list = obs.set_index(index).columns.values
    X_name_list = fea.set_index(index).columns.values[mask]
    
    return y, X, y_name_list, X_name_list


def grid_search(Yendog_train, Yendog_val, exog_train, exog_val, cfg_list, n_jobs=False, parallel=True, ts_index=''):
    """
            Learn with all the parameters configuration in cfg_list and return scores for each of them.
    """
    if n_jobs==False:
        n_jobs = cpu_count()
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=n_jobs, backend='multiprocessing')
        tasks = (delayed(score_model)(Yendog_train, Yendog_val, exog_train, exog_val, cfg, debug=False) for cfg in tqdm(cfg_list,desc='Time series {}'.format(ts_index)))
        scores = executor(tasks)
    else:
        scores = [score_model(Yendog_train, Yendog_val, exog_train, exog_val, cfg) for cfg in tqdm(cfg_list, desc='Time series {}'.format(ts_index))]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores

def score_model(Yendog_train, Yendog_val, exog_train, exog_val, cfg, debug=False):
    """
            Fit model with parameters configuration cfg and return score with debug mode or not.
    """
    result = None
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        predictions = fit_and_predict(Yendog_train, exog_train, exog_val, cfg)
        error = measure_rmse(Yendog_val.flatten(), predictions)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                predictions = fit_and_predict(Yendog_train, exog_train, exog_val, cfg)
                error = measure_rmse(Yendog_val.flatten(), predictions)
        except:
            error = None
    # check for an interesting result
    #if result is not None:
        #print(' > Model[{}] {:.2f}'.format(key, result))
    return (key, error)

def fit_and_predict(Yendog_train, exog_train, exog_val, cfg):
    """
            Fit model with parameters configuration cfg and return error RMSE
    """
    order, sorder, trend = cfg
    model = sarimax.SARIMAX(Yendog_train.flatten(), exog_train, order=order, seasonal_order=sorder, trend=trend,
                            enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    predictions = model_fit.predict(start=len(Yendog_train), end=len(Yendog_train)+len(exog_val)-1, exog=exog_val)
    #plt.plot(predictions, label ='pred')
    #plt.plot(Yendog_val.flatten(), label ='obs')
    #plt.legend()
    #plt.show()


    return predictions


def sarima_configs(param_grid):
    """
            Create list of parameters configuration from a parameters grid.
    """
    configs = [[(i[0],i[1],i[2]), (i[3],i[4],i[5],i[6]), i[7]] for i in itertools.product(*[param_grid['p_params'],
                                                                                            param_grid['d_params'],
                                                                                            param_grid['q_params'],
                                                                                            param_grid['P_params'],
                                                                                            param_grid['D_params'],
                                                                                            param_grid['Q_params'],
                                                                                            param_grid['m_params'],
                                                                                            param_grid['t_params']])]
    return configs



def measure_rmse(obs, pred):
    return sqrt(mean_squared_error(obs, pred))