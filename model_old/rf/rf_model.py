"""
        Random Forest is a machine learning model..
        Input: Observation per date of each time-series, exogenous data
"""


import os
import sys


sys.path.insert(0, '../../')
from model.forecast_model import Forecast_model

import numpy as np
from tqdm import tqdm
from shutil import copyfile
import pandas as pd
from scipy import spatial

try:
    import cPickle as pickle
except:
    import pickle


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

class Rf_model(Forecast_model):

    def __init__(self, name: str, start_date: str, end_date: str, features_list, time_series, observation_data_path,
                 exogenous_data_path, scaler_choice: str):
        """
                Initialisation of the model Historical average
                Datetime format 'YYYY-MM-dd hh-mm-ss'

                :param name: name of the model
                :param start_date: inclusive (e.g. 2015-01-01 00:00:00)
                :param end_date: inclusive (e.g. 2016-12-31 23:45:00)
                :param features_list: features_list of the exogenous data to take into account by the model (e.g., day of the week, ...)
                :param time_series: time_series selected for learning
                :param observation_data_path: list of file containing the data of observation (header: Datetime, id_time_series1, id_time_series2,...)
                :param exogenous_data_path: list of file containing the data of date (header: Datetime, feature1, feature2,...)
                :param scaler_choice: choice of scaler (minmax or standard)
        """

        Forecast_model.__init__(self, name)
        self.infos['start_date'] = start_date
        self.infos['end_date'] = end_date
        self.infos['features_list'] = features_list
        self.infos['time_series'] = time_series
        self.infos['observation_data_path'] = observation_data_path
        self.infos['exogenous_data_path'] = exogenous_data_path
        self.infos['scaler_choice'] = scaler_choice

        scaler_choices = ['minmax', 'standard', None]
        if scaler_choice not in scaler_choices:
            raise ValueError("Invalid scaler choice. Expected one of: {}".format(scaler_choices))


    def __str__(self):
        return "Description of model: %s\n" \
               "Learning start date: %s\n" \
               "Learning end date: %s\n" \
               "Features list: %s\n" \
               "Learned time series: %s\n" \
               "Training observation data path: %s\n" \
               "Training exogenous data (date) path: %s" \
               "Scaler choice: %s\n" % (self.name, self.infos.start_date, self.infos.end_date,
                                                            str(self.infos.features_list),
                                                            ",".join(str(x) for x in self.infos.time_series),
                                                            self.infos.observation_data_path, self.infos.exogenous_data_path,
                                                            self.infos.scaler_choice)

    def save(self, path_directory_to_save, path_config_file):
        """
                Save the model in a pickle and the configuration yaml file in directory 'path_directory_to_save'

        """
        if not os.path.exists(path_directory_to_save):
            os.makedirs(path_directory_to_save)

        with open(path_directory_to_save+self.infos['name']+'.pkl', 'wb') as pickle_file:
            pickle.dump(self.infos, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        if path_config_file != None:
            copyfile(path_config_file, path_directory_to_save+'/'+path_config_file.split('/')[-1])


    def optimize(self, X_list, y, param_grid, param_kfold, scoring):

        grid_search_dict = {}

        for features, X in zip(self.infos['features_list'], X_list):

            if self.infos['scaler_choice'] == 'minmax':
                scaler = MinMaxScaler(feature_range=(0, 1))
                X = scaler.fit_transform(X)
            elif self.infos['scaler_choice'] == 'standard':
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            rf = RandomForestRegressor()
            cv = KFold(**param_kfold)
            grid_search = GridSearchCV(rf, param_grid=param_grid, n_jobs=1, cv=cv, verbose=1, scoring=scoring)
            grid_search.fit(X, y)

            # Save space disk
            grid_search.estimator = None
            grid_search.best_estimator_ = None
            grid_search_dict[tuple(features)] = grid_search

        return grid_search_dict

    # Utility function to report best scores
    # def report(results, n_top=3):
    #    for i in range(1, n_top + 1):
    #        candidates = np.flatnonzero(results['rank_test_score'] == i)
    #        for candidate in candidates:
    #            print("Model with rank: {0}".format(i))
    #            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
    #                  results['mean_test_score'][candidate],
    #                  results['std_test_score'][candidate]))
    #            print("Parameters: {0}".format(results['params'][candidate]))
    #            print("")
    # start=time()
    # print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
    #      % (time() - start, len(grid_search.cv_results_['params'])))
    # report(grid_search.cv_results_,n_top=3)

    def fit(self, X, y, best_params):
        """
                Learn Random Forest model with with best params.
                :param X:
                :param y:
                :param best_params:
                :return:
        """
        if self.infos['scaler_choice'] == 'minmax':
            scaler = MinMaxScaler(feature_range=(0, 1))
            X = scaler.fit_transform(X)
        elif self.infos['scaler_choice'] == 'standard':
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        self.infos['scaler'] = scaler

        rf = RandomForestRegressor(**best_params)
        rf.fit(X,y)

        return rf


    def predict(self, rf, X):
        if (self.infos['scaler_choice'] == 'minmax') or (self.infos['scaler_choice'] == 'standard'):
            x = self.infos['scaler'].transform(X)
        else:
            x = X

        return rf.predict(x)
