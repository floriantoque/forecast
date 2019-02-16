"""
        The historical average model is a baseline forecasting model used for long-term forecasting.
        The prediction of this model are based on the average of the observation depending on the selected features that
        depict the date (e.g., day, timestep, holiday).
        During the prediction of new data, if one of the feature tuple that depict a date
        is not in the database, the most similar tuple of the database is selected for the prediction.

        Input: Observation per date of each time-series, exogenous data that depicts the date
"""

import sys

sys.path.insert(0, '../')
from forecast_model import Forecast_model

import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
from scipy import spatial

try:
    import cPickle as pickle
except:
    import pickle

sys.path.insert(0, '../../')
from utils.utils import *
from utils.utils_date import *


class Ha_inverted_model(Forecast_model):

    def __init__(self, name: str):
        Forecast_model.__init__(self, name)
        self.infos['start_date'] = ''
        self.infos['end_date'] = ''
        self.infos['features'] = []
        self.infos['features_day'] = []
        self.infos['time_series'] = []
        self.infos['observation_path'] = ''
        self.infos['context_path'] = ''

    def __str__(self):
        return "Description of model: %s\n" \
               "Start date learing: %s\n" \
               "End date learing: %s\n" \
               "Features: %s\n" \
               "Features day: %s\n" \
               "Learned time series: %s\n" \
               "Training observation path: %s\n" \
               "Training context data path: %s" % (self.infos['name'],
                                                   self.infos['start_date'],
                                                   self.infos['end_date'],
                                                   self.infos['features'],
                                                   self.infos['features_day'],
                                                   self.infos['time_series'],
                                                   self.infos['observation_path'],
                                                   self.infos['context_path'])

    def fit(self, X, ylist):

        dict_pred_mean = {}
        dict_pred_median = {}
        for ix, ts in enumerate(tqdm(self.infos['time_series'])):
            data = np.concatenate([X, ylist[ix]], axis=1)
            dfXy = pd.DataFrame(data=data, columns=np.arange(data.shape[1]).astype(str))
            df_mean = dfXy.groupby(dfXy.columns.values[:X.shape[1]].tolist()).mean()
            df_median = dfXy.groupby(dfXy.columns.values[:X.shape[1]].tolist()).median()

            for f in df_mean.index.values:
                try:
                    dict_pred_mean[tuple(f)][ts] = df_mean.loc[f].values.astype(float)
                    dict_pred_median[tuple(f)][ts] = df_median.loc[f].values.astype(float)
                except:
                    dict_pred_mean[tuple(f)] = {}
                    dict_pred_mean[tuple(f)][ts] = df_mean.loc[f].values.astype(float)
                    dict_pred_median[tuple(f)] = {}
                    dict_pred_median[tuple(f)][ts] = df_median.loc[f].values.astype(float)
        self.infos['dict_pred_mean'] = dict_pred_mean
        self.infos['dict_pred_median'] = dict_pred_median
        return

    def predict(self, X, choice='mean'):
        if choice == 'mean':
            dict_pred = self.infos['dict_pred_mean']
        else:
            dict_pred = self.infos['dict_pred_median']

        possibles_day = [i for i in list(set(dict_pred.keys()))]
        pred_all = []
        for x in tqdm(X):
            pred = []
            try:
                for ts in self.infos['time_series']:
                    pred.append(dict_pred[tuple(x)][ts])
            except:
                npd = nearest_tuple(possibles_day, tuple(x))
                for ts in self.infos['time_series']:
                    pred.append(dict_pred[npd][ts])
            pred_all.append(pred)
        return np.array(pred_all)
