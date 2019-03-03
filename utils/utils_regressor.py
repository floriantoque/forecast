from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.insert(0, '../')
import model.historical_average.ha_model as ha_model

from tqdm import tqdm

import numpy as np



def scaler_fit_transform(x, scaler_choice=None):
    if len(x.shape)<=2:
        if scaler_choice == 'standard':
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
        elif scaler_choice == 'minmax':
            scaler = MinMaxScaler()
            x = scaler.fit_transform(x)
        else:
            scaler = None
    else:
        scaler_list = []
        x_list = []
        if scaler_choice == 'standard':
            for i in x:
                scaler = StandardScaler()
                x_ = scaler.fit_transform(i)
                scaler_list.append(scaler)
                x_list.append(x_)
            x = np.array(x_list)
            scaler = scaler_list
        elif scaler_choice == 'minmax':
            for i in x:
                scaler = MinMaxScaler()
                x_ = scaler.fit_transform(i)
                scaler_list.append(scaler)
                x_list.append(x_)
            x = np.array(x_list)
            scaler = scaler_list
        else:
            scaler = None
    return x, scaler

def scaler_transform(x, scaler=None):
    if scaler == None:
        return x
    else:
        x = scaler.transform(x)
        return x

def scaler_inverse_transform(x, scaler=None):
    if scaler == None:
        return x
    elif type(scaler) != list:
        x = scaler.inverse_transform(x)
        return x
    else:
        x = np.array([s.inverse_transform(x[i]) for i, s in enumerate(scaler)])
        return x
            

def get_estimator(estimator_choice = 'rf', params=None, n_jobs=1):
    if estimator_choice not in ['rf', 'lr', 'gp', 'en', 'ha', 'svr', 'gbr']:
        print('Error: estimator_choice value not good')
    if estimator_choice == 'rf':
        if params==None:
            estimator = RandomForestRegressor(n_jobs=n_jobs)
        else:
            params['n_jobs'] = n_jobs
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
    if estimator_choice == 'en':
        if params==None:
            estimator = ElasticNet()
        else:
            estimator = ElasticNet(**params)
    if estimator_choice == 'ha':
        if params==None:
            estimator = ha_model.Ha_model()
        else:
            estimator = ha_model.Ha_model(**params)
    if estimator_choice == 'svr':
        if params==None:
            estimator = MultiOutputRegressor(SVR(), n_jobs=n_jobs)
        else:
            try:
                estimator = MultiOutputRegressor(SVR(**params), n_jobs=n_jobs)
            except:
                params = {k.split('estimator__')[1]: params[k] for k in list(params.keys())}
                estimator = MultiOutputRegressor(SVR(**params), n_jobs=n_jobs)
    if estimator_choice == 'gbr':
        if params==None:
            estimator = MultiOutputRegressor(GradientBoostingRegressor(),
                                             n_jobs=n_jobs)
        else:
            try:
                estimator = MultiOutputRegressor(GradientBoostingRegressor(**params),
                                                 n_jobs=n_jobs)
            except:
                params = {k.split('estimator__')[1]: params[k] for k in
                          list(params.keys())}
                estimator = MultiOutputRegressor(GradientBoostingRegressor(**params),
                                                 n_jobs=n_jobs)
    return estimator

def optimize_multioutput_regressor_multiseries_model(X, y_list, param_grid, param_kfold, estimator, verbose=0,
                                                     n_jobs_cv=-1):
    cv = KFold(**param_kfold)
    
    best_params_time_series = {}
    cv_results_time_series = {}

    for ydx, y in enumerate(tqdm(y_list, desc="Optimization: loop over time-series")):
        try:
            grid_search_res = GridSearchCV(estimator, param_grid, cv=cv,
                                           verbose=verbose, n_jobs=n_jobs_cv,
                                           return_train_score=True)
            grid_search_res = grid_search_res.fit(X, y,)
        except ValueError:
            print("ValueError: param_grid may do no fit with the estimator.")
            break
        cv_results_time_series[ydx] = {}
        best_params_time_series[ydx] = {}
        cv_results_time_series[ydx]['cv_results_'] = grid_search_res.cv_results_
        best_params_time_series[ydx]['best_params_'] = grid_search_res.best_params_
 
    return best_params_time_series, cv_results_time_series

    
def fit_multioutput_regressor_multiseries_model(X, y_list, estimator_choice,
                                                best_params_time_series=None,
                                                n_jobs=1):
    estimator_time_series = {}
    if best_params_time_series == None:
        for ydx, y in enumerate(tqdm(y_list,desc='Fit: loop over time-series')):
            estimator = get_estimator(estimator_choice, n_jobs=n_jobs)
            estimator = estimator.fit(X, y)
            estimator_time_series[ydx] = estimator
    else:
        for ydx, y in enumerate(tqdm(y_list,desc='Fit: loop over time-series')):
            estimator = get_estimator(estimator_choice,
                                      best_params_time_series[ydx]['best_params_'],
                                      n_jobs=n_jobs)
            estimator = estimator.fit(X, y)
            estimator_time_series[ydx] = estimator
    return estimator_time_series

def fit_multioutput_regressor_uniseries_model(X, y, estimator_choice,
                                                best_params_=None,
                                                n_jobs=1):
    if best_params_ == None:
        estimator = get_estimator(estimator_choice, n_jobs=n_jobs)
        estimator = estimator.fit(X, y)
    else:
        estimator = get_estimator(estimator_choice,
                                  best_params_,
                                  n_jobs=n_jobs)
        estimator = estimator.fit(X, y)
    return estimator

def predict_multioutput_regressor_multiseries_model(X, estimator_time_series):
    pred = []
    for i,k in enumerate(tqdm(estimator_time_series.keys(), desc='Predict: loop over time-series')):
        pred.append(estimator_time_series[i].predict(X))
    pred = np.array(pred)
    #pred = pred.reshape(pred.shape[0], pred.shape[1]*pred.shape[2])
    pred[pred<0] = 0
    return pred

def predict_multioutput_regressor_uniseries_model(X, estimator):
    pred = estimator.predict(X)
    #pred = pred.reshape(pred.shape[0]*pred.shape[1])
    pred[pred<0] = 0
    return pred
