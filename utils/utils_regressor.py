from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from tqdm import tqdm

import numpy as np

def get_estimator(estimator_choice = 'rf', params=None):
    if estimator_choice not in ['rf', 'lr', 'gp']:
        print('Error: estimator_choice value not good')
    if estimator_choice == 'rf':
        if params==None:
            estimator = RandomForestRegressor()
        else:
            estimator = RandomForestRegressor(**params)
    if estimator_choice == 'lr':
        if params==None:
            estimator = LinearRegression()
        else:
            estimator = LinearRegression(**params)
    if estimator_choice == 'gp':
        if params==None:
            estimator = GaussianProcessRegressor()
        else:
            estimator = GaussianProcessRegressor(**params)
    return estimator


def optimize_multioutput_regressor_multiseries_model(X, y_list, param_grid, param_kfold, estimator, verbose=0,
                                                     n_jobs=-1):
    cv = KFold(**param_kfold)
    
    grid_search_time_series = {}

    for ydx, y in enumerate(tqdm(y_list, desc="Optimization: loop over time-series")):
        try:
            grid_search_res = GridSearchCV(estimator, param_grid, cv=cv, verbose=verbose, n_jobs=n_jobs)
            grid_search_res = grid_search_res.fit(X, y,)
        except ValueError:
            print("ValueError: param_grid may do no fit with the estimator.")
            break

        grid_search = {'cv_results_': grid_search_res.cv_results_,'best_params_': grid_search_res.best_params_ }
        grid_search_time_series[ydx] = grid_search
 
    return grid_search_time_series
       
    
def fit_multioutput_regressor_multiseries_model(X, y_list, estimator_choice, best_params_time_series):
    estimator_time_series = {}
    for ydx, y in enumerate(tqdm(y_list,desc='Fit: loop over time-series')):
        estimator = get_estimator(estimator_choice, best_params_time_series[ydx]['best_params_'])
        estimator = estimator.fit(X, y)
        estimator_time_series[ydx] = estimator
    return estimator_time_series


def predict_multioutput_regressor_multiseries_model(X, estimator_time_series):
    pred = []
    for i,k in enumerate(tqdm(estimator_time_series.keys(), desc='Predict: loop over time-series')):
        pred.append(estimator_time_series[i].predict(X))
    
    pred = np.array(pred)
    pred = pred.reshape(pred.shape[0], pred.shape[1]*pred.shape[2])
    pred[pred<0] = 0
    
    return pred


