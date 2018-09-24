"""
        The historical average model is a baseline forecasting model used for long-term forecasting.
        The prediction of this model are based on the average of the observation depending on the selected features that
        depict the date (e.g., day, timestep, holiday).
        During the prediction of new data, if one of the feature tuple that depict a date
        is not in the database, the most similar tuple of the database is selected for the prediction.

        Input: Observation per date of each time-series, exogenous data that depicts the date
"""



import sys

sys.path.insert(0, '../../')
from model.forecast_model import Forecast_model
from utils.utils import *

import numpy as np
from tqdm import tqdm
import os
from shutil import copyfile
import pandas as pd


try:
    import cPickle as pickle
except:
    import pickle




class Ha_model(Forecast_model):

    def __init__(self, name: str, start_date: str, end_date: str, features, time_series, observation_data_path: str, exogenous_data_path: str):
        """
                Initialisation of the model Historical average
                Datetime format 'YYYY-MM-dd hh-mm-ss'

                :param name: name of the model
                :param start_date: inclusive (e.g. 2015-01-01 00:00:00)
                :param end_date: inclusive (e.g. 2016-12-31 23:45:00)
                :param features: features of the date take into account by the model (e.g., day of the week, ...)
                :param time_series: time_series selected for learning
                :param observation_data_path: file containing the data of observation (header: Datetime, id_time_series1, id_time_series2,...)
                :param exogenous_data_path: file containing the data of date (header: Datetime, feature1, feature2,...)
        """

        Forecast_model.__init__(self, name)
        self.infos['start_date'] = start_date
        self.infos['end_date'] = end_date
        self.infos['features'] = features
        self.infos['time_series'] = time_series
        self.infos['observation_data_path'] = observation_data_path
        self.infos['exogenous_data_path'] = exogenous_data_path


    def __str__(self):
        return "Description of model: %s\n" \
               "Learning start date: %s\n" \
               "Learning end date: %s\n" \
               "Features: %s\n" \
               "Learned time series: %s\n" \
               "Training observation data path: %s\n" \
               "Training exogenous data (date) path: %s" % (self.infos['name'], self.infos['start_date'], self.infos['end_date'],
                                                            ",".join(str(x) for x in self.infos['features']),
                                                            ",".join(str(x) for x in self.infos['time_series']),
                                                            self.infos['observation_data_path'], self.infos['exogenous_data_path'])




    def fit(self, X, y):
        """
                Learn the historical average model
                :param X: list of list or array of list with features values
                :param y: list of list or array of list with time_series values
                :return:
        """

        # if not all([isinstance(el, list) for el in y]):
        #    #Check if all element of y are of type list, if not create list of list of element
        #    y = [[i] for i in y]

        data = [ [tuple(ft)]+list(ts) for ft,ts in zip(list(X),list(y))]
        columns = ['features_tuple'] + self.infos['time_series']

        df = pd.DataFrame(data=data, columns=columns)

        print('Progress Bar 1/2 : learning mean')
        self.infos['dict_pred_mean'] =   dict([ (i[0], np.array(i[1:]).astype(float)) for i in tqdm(df.groupby('features_tuple').mean().reset_index().values)] )
        print('Progress Bar 2/2 : learning median')
        self.infos['dict_pred_median'] = dict([ (i[0], np.array(i[1:]).astype(float)) for i in tqdm(df.groupby('features_tuple').median().reset_index().values)] )

        return

    def save(self, path_directory_to_save, path_config_file):
        """
                Save the historical average model in a pickle and the configuration yaml file in directory 'path_directory_to_save'

        """
        if not os.path.exists(path_directory_to_save):
            os.makedirs(path_directory_to_save)

        with open(path_directory_to_save+self.infos['name']+'.pkl', 'wb') as pickle_file:
            pickle.dump(self, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        if path_config_file != None:
            copyfile(path_config_file, path_directory_to_save+'/'+path_config_file.split('/')[-1])


    # def predict_old(self, list_date, df_exogenous, choice='mean'):
    #     """
    #             Predict each time-series with the fitted Historical average model.
    #             :param list_date: List of date (Datetime format 'YYYY-MM-dd hh-mm-ss')
    #             :param df_exogenous: Dataframe with header (Datetime, feature1, feature2, ...)
    #             :param choice: Historical average model (use the mean or median of observation)
    #             :return: Dataframe with header (Datetime, time-series1, time-series2, ...) and data of the HA model prediction
    #     """
    #
    #     X = df_exogenous.set_index('Datetime').ix[list_date][self.infos['features']]
    #     info_to_pred = [tuple(i) for i in  X.values.tolist()]
    #
    #     possibles_day = [i for i in list(set(self.infos['dict_pred_mean'].keys()))]
    #
    #     df_observation = pd.DataFrame(data = list_date, columns=["Datetime"])
    #     for s in self.infos['time_series']:
    #         df_observation[s] = np.nan
    #
    #
    #     res = []
    #
    #     if choice != 'mean':
    #         choice = 'median'
    #
    #     for inf in tqdm(info_to_pred):
    #         try:
    #             res.append(self.infos['dict_pred_'+choice][inf])
    #         except:
    #             npd = nearest_tuple(possibles_day, inf)
    #             res.append(self.infos['dict_pred_'+choice][npd])
    #     res = np.array(res)
    #
    #     df_res = pd.DataFrame()
    #     df_res["Datetime"] = df_observation["Datetime"].tolist()
    #
    #     for i, s in enumerate(self.infos['time_series']):
    #         df_res[s] = res[:, i]
    #
    #     return df_res


    def predict(self, X, choice='mean'):
        """
                Predict each time-series with the fitted Historical average model.
                :param list_date: List of date (Datetime format 'YYYY-MM-dd hh-mm-ss')
                :param df_exogenous: Dataframe with header (Datetime, feature1, feature2, ...)
                :param choice: Historical average model (use the mean or median of observation)
                :return: Dataframe with header (Datetime, time-series1, time-series2, ...) and data of the HA model prediction
        """


        #info_to_pred = [tuple(i) for i in  list(X) ]

        possibles_day = [i for i in list(set(self.infos['dict_pred_mean'].keys()))]

        pred = []

        if choice != 'mean':
            choice = 'median'

        for i in tqdm(list(X)):

            try:
                pred.append(np.array(self.infos['dict_pred_'+choice][tuple(i)]).astype(float))
            except:
                npd = nearest_tuple(possibles_day, tuple(i))
                pred.append(np.array(self.infos['dict_pred_'+choice][npd]).astype(float))

        pred = np.array(pred)


        return pred

