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

import utils.utils as utils
import utils.utils_date as utils_date
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
from scipy import spatial



class Ha_model:

    def __init__(self, method='mean'):
        if method not in ['mean', 'median']:
            raise ValueError('{method} wrong, use "mean" or "median"'.format(method=repr(method)))
        else:
            self.dict_fea_pred = {}
            self.method = method
 

    def fit(self, X, y):
        data = np.concatenate([X, y], axis=1)
        
        dfXy = pd.DataFrame(data=data, columns=np.arange(data.shape[1]).astype(str))
        
        if self.method == 'mean':
            df = dfXy.groupby(dfXy.columns.values[:X.shape[1]].tolist()).mean()
            for f in df.index.values:
                self.dict_fea_pred[tuple(f)] = df.loc[f].values.astype(float)

        if self.method == 'median':
            df = dfXy.groupby(dfXy.columns.values[:X.shape[1]].tolist()).median()
            for f in df.index.values:
                self.dict_fea_pred[tuple(f)] = df.loc[f].values.astype(float)
            
        return self

    def predict(self, X):

        features = [i for i in list(set(self.dict_fea_pred.keys()))]
        pred_all = []
        for x in tqdm(X):
            try:
                pred_all.append(self.dict_fea_pred[tuple(x)])
            except:
                nearest_features = utils.nearest_tuple(features, tuple(x))
                pred_all.append(self.dict_fea_pred[nearest_features])
        return np.array(pred_all)
