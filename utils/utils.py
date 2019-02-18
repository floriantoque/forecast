try:
    import cPickle as pickle
except:
    import pickle

import os
import numpy as np
import pandas as pd
import copy

from scipy import spatial

from utils.utils_date import *
from tqdm import tqdm


def yes_or_no(question):
    """
            Ask to the user a question. Possible answer y (yes) or n (no)
            :param question: str
            :return: True or False depending on the choice of the user
    """
    while "The answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False

def save_pickle(path, obj):

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'wb') as pickle_file:
        pickle.dump(obj, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    return


def df_to_csv_safe(path, df, force_save=False):
    if force_save:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        df.to_csv(path, index=False)
    elif os.path.isfile(path):
        if yes_or_no("The file {} already exist. Do you want to erase it?".format(path)):
            df.to_csv(path, index=False)
    else:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        df.to_csv(path, index=False)

def save_pickle_safe(path, obj, force_save=False):
    """
            Save obj in pickle file but ask question if the file already exists.
    """
    if force_save:
        save_pickle(path,obj)
    elif os.path.isfile(path):
        if yes_or_no("The file {} already exist. Do you want to erase it?".format(path)):
            save_pickle(path, obj)
    else:
        save_pickle(path, obj)

def load_pickle(path):
    """
            Read a pickle file and load the pickle element
            :param path: path of the pickle file
            :return: element in the pickle file
    """
    with open(path, 'rb') as filename:
        return pickle.load(filename)


def size_of_object(obj):
    """
            Calculate the memory space used on the disk of an object saved in pickle
            !!! Does not work with numpy arrays, ...
            :param obj: object
            :return: size in bytes of the object
    """
    size = len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    return size

def read_csv_list(csv_list, index_ = 'Datetime'):
    """
            :param csv_list: list of csv files
            :param index_: index_ used for the join
            :return: dataframe that join on index_ all the csv files
    """
    df = pd.read_csv(csv_list[0]).set_index(index_)
    df = df.join([pd.read_csv(i).set_index(index_) for i in csv_list[1:]])
    df = df.reset_index()
    df = df.drop_duplicates()
    return df




def nearest_tuple(tuple_list, tuple_):
    """
            Compute the cosine similarity between tuples in tuple list and tuple  and select the nearest tuple

            :param tuple_list: list of tuple of integers
            :param tuple_: tuple of integers
            :return: nearest tuple between possibles_day and info
    """
    list_ = [((1 - spatial.distance.cosine(i, tuple_)), i) for i in tuple_list]
    return tuple(sorted(list_, key=lambda x: x[0], reverse=True)[0][1])


def window_Xy(df_Xy, time_series, features, features_window, tminus, tplus):
    """
            return X , y and fname (features name with lag window e.g., tminus=2 , tplus=0, f1 = [f1::t-2, f1::t-1,f1] )
            :param df_Xy: dataframe with exogenous and observation value
            :param time_series: used to create y
            :param features: used as exogenous data
            :param features_window: used as exogenous windowed data
            :param tminus: window tminus (tO is not included)
            :param tplus: window tplus (t0 is non included)
            :return: X, y, fname
    """
    y = df_Xy[time_series].values[tminus:len(df_Xy) - tplus, :]
    X = np.zeros((len(df_Xy) - (tminus + tplus), len(features) + len(features_window) * (tplus + tminus)))

    cpt = 0
    fname = []
    for f in features:
        if not f in features_window:
            X[:, cpt] = df_Xy[f].values[tminus:len(df_Xy) - tplus]
            cpt += 1
            fname.append(f)
        else:
            X[:, cpt:cpt + tminus + tplus + 1] = np.array(
                [df_Xy[f].values[i - tminus:i + (tplus + 1)] for i in range(tminus, len(df_Xy) - (tplus))])
            cpt += tminus + tplus + 1
            fname += [f + '::t-' + str(i) for i in range(1, tminus + 1)[::-1]] + [f] + [f + '::t+' + str(i) for i in
                                                                                        range(1, tplus + 1)]

    return X, y, fname

def window_X(df_X, features, features_window, tminus, tplus):
    """
            return X and fname (features name with lag window e.g., tminus=2 , tplus=0, f1 = [f1::t-2, f1::t-1,f1] )
            :param df_X: dataframe with exogenous
            :param features: used as exogenous data
            :param features_window: used as exogenous windowed data
            :param tminus: window tminus (tO is not included)
            :param tplus: window tplus (t0 is non included)
            :return: X, fname
    """

    X = np.zeros((len(df_X) - (tminus + tplus), len(features) + len(features_window) * (tplus + tminus)))

    cpt = 0
    fname = []
    for f in features:
        if not f in features_window:
            X[:, cpt] = df_X[f].values[tminus:len(df_X) - tplus]
            cpt += 1
            fname.append(f)
        else:
            X[:, cpt:cpt + tminus + tplus + 1] = np.array(
                [df_X[f].values[i - tminus:i + (tplus + 1)] for i in range(tminus, len(df_X) - (tplus))])
            cpt += tminus + tplus + 1
            fname += [f + '::t-' + str(i) for i in range(1, tminus + 1)[::-1]] + [f] + [f + '::t+' + str(i) for i in
                                                                                        range(1, tplus + 1)]

    return X, fname


AT = 5

def mse(obs, pred):
    return ((pred - obs) ** 2).mean()


def rmse(obs, pred):
    return np.sqrt(mse(obs, pred))


def mae(obs, pred):
    return np.absolute(pred - obs).mean()


def mape_at(obs, pred):
    mask = obs >= AT
    return ((np.absolute(pred[mask] - obs[mask]) / obs[mask]).mean())*100


def build_Xylist(df, features, features_day, time_series, start_date, end_date):
    """

    :param df: dataframe that contains time-series, features and features_day in header and Datetime (timestep)
    :param features: one features per day
    :param features_day: features all along the day
    :param time_series:
    :param start_date:
    :param end_date:
    :return:
    """
    # Build ylist
    days = sorted(list(set([i[:10] + ' 00:00:00' for i in df['Datetime'].values])))
    df_observation = copy.deepcopy(df[['Datetime'] + time_series])
    df_observation['time'] = [d[11:] for d in df_observation['Datetime'].values]
    df_observation['Datetime'] = [d[:10] + ' 00:00:00' for d in df_observation['Datetime'].values]
    df_observation = df_observation.pivot_table(values=time_series, index='Datetime', columns='time')
    ylist = np.array([df_observation[ts].values for ts in time_series])

    # Build X
    if len(features_day) > 0:
        df_context_day = copy.deepcopy(df[['Datetime'] + features_day])
        df_context_day['time'] = [d[11:] for d in df_context_day['Datetime'].values]
        df_context_day['Datetime'] = [d[:10] + ' 00:00:00' for d in df_context_day['Datetime'].values]
        df_context_day = df_context_day.pivot_table(values=features_day, index='Datetime', columns='time')
        df_context_day.columns = df_context_day.columns.map('|'.join)
        dfX = df_context_day.join(df.set_index('Datetime')[features].loc[days])
    else:
        dfX = df.set_index('Datetime')[features].loc[days]
    features_end = dfX.columns.values.tolist()

    X = dfX.values
    return X, ylist, features_end


def build_X(df, features, features_day, time_series, start_date, end_date):
    # Build X
    days = sorted(list(set([i[:10] + ' 00:00:00' for i in df['Datetime'].values])))
    if len(features_day) > 0:
        df_context_day = copy.deepcopy(df[['Datetime'] + features_day])
        df_context_day['time'] = [d[11:] for d in df_context_day['Datetime'].values]
        df_context_day['Datetime'] = [d[:10] + ' 00:00:00' for d in df_context_day['Datetime'].values]
        df_context_day = df_context_day.pivot_table(values=features_day, index='Datetime', columns='time')
        df_context_day.columns = df_context_day.columns.map('|'.join)
        days = df_context_day.index.values
        dfX = df_context_day.join(df.set_index('Datetime')[features].loc[days])
    else:
        dfX = df.set_index('Datetime')[features].loc[days]
    features_end = dfX.columns.values.tolist()

    X = dfX.values
    return X, features_end


def df_todummy_df(df, features_todummy):
    """
            In dataframe df, create dummy variables of columns in features_todummy
    """
    df_= df.copy()
    for f in features_todummy:
        d = pd.get_dummies(df_[f])
        df_ = pd.concat([df_, d], axis=1)
        df_.drop([f, d.columns.values[-1]], inplace=True, axis=1)
    df_.columns = df_.columns.values.astype(str)
    return df_


def create_dfXy_long_term0(df_obs, df_fea, time_series, features_time_step, features_day, features_todummy=[], index='Datetime',
                start_time='00:00:00', end_time='23:45:00', time_step_second=15*60):
    '''
        Create dfXy for long-term forecasting method (0), with features_time_step and features_day
        features_time_step have to be full over all the time step
        features_day can be full only on days (YYYY-mm-dd 00:00:00) exemple: 
        '2015-01-01 00:00:00': fea1 fea2
        '2015-01-01 01:00:00': Nan Nan
        or
        '2015-01-01 00:00:00': fea1 fea2
        '2015-01-02 00:00:00': fea1 fea2
    '''
    
    df_obs_ = df_obs[[index] + time_series].copy()
    df_fea_ = df_fea[[index] + features_time_step + features_day].copy()
    
    # Fill time step of df_obs_
    days = sorted(list(set([i[:10] for i in df_obs_[index].values])))
    timestamp_list = [j for i in [build_timestamp_list(d+' '+start_time, d+' '+end_time, time_step_second=time_step_second) for d in days] for j in i]
    df_date = pd.DataFrame(data = timestamp_list, columns = [index]).set_index(index)
    df_obs_ = df_date.join(df_obs_.set_index(index)).fillna(0)
    
    # Fill time step of df_fea_[features_day]
    days = sorted(list(set([i[:10] for i in df_fea_[index].values])))
    timestamp_list = [j for i in [build_timestamp_list(d+' '+start_time, d+' '+end_time, time_step_second=time_step_second) for d in days] for j in i]
    df_date = pd.DataFrame(data = timestamp_list, columns = [index]).set_index(index)
    df_fea_1 = df_date.join(df_fea_.set_index(index)[features_day]).fillna(method='ffill').fillna(method='bfill')
    df_fea_ = df_fea_.set_index(index)[features_time_step].join(df_fea_1).reset_index()
    
    # Features to dummy df_fea_
    df_fea_ = df_todummy_df(df_fea_, features_todummy)
    
    # Get features_time_step_name
    fea_timestep_todummy = [i for i in features_time_step if i in features_todummy]
    features_time_step_name = [i for i in features_time_step if i not in features_todummy]
    fea_timestep_todummy_name = []
    for f in fea_timestep_todummy:
        fea_timestep_todummy_name += pd.get_dummies(df_fea[f]).columns.values.astype(str).tolist()[:-1]
    features_time_step_name += fea_timestep_todummy_name
        
    return df_obs_.join(df_fea_.set_index(index)).dropna().reset_index(), features_time_step_name


def create_xy_dataset_long_term0(dfXy, time_series, features_time_step_name, index='Datetime', start_time='00:00:00', end_time='23:45:00', nb_time_step = 24*4, reduce=False):    
    '''
        Create X,y,X_names,days for long-term foreasting method (0)
        Use dfXy obtained with function create_dfXy_long_term0
    '''
   
    dfXy_ = dfXy.set_index(index).copy()
    
    days = sorted(list(set([i[:10] for i in dfXy_.index.values])))
    other_features = [i for i in dfXy_.columns.tolist() if i not in time_series + features_time_step_name]
    X_names = [f+'-T'+str(ix) for f in features_time_step_name for ix in np.arange(nb_time_step)] + other_features
    
    X = []
    list_y=[]
    for d in tqdm(days, desc='Days loop'):
        df_timestep = dfXy_.loc[d+' '+start_time: d+' '+end_time][features_time_step_name].values.T.flatten()
        df_day = dfXy_.loc[[d+' '+start_time]][other_features].values.flatten()
        X.append(np.concatenate([df_timestep, df_day]))
        y = []
        for s in time_series:
            y.append(dfXy_.loc[d+' '+start_time: d+' '+end_time][s].values)
        list_y.append(y)
    
    X = np.array(X)
    y_list = np.swapaxes(np.array(list_y),0,1)
    X_names = np.array(X_names)
    
    if reduce:
        mask = np.array([(x[0]!=x[1:]).sum()!=0 for x in X.T])
        X = X[:,mask]
        X_names = X_names[mask]
    
    return X, y_list, X_names, days


def pred_day_array_to_df(pred, time_series, days, time_step_second=15*60, index='Datetime', start_time='00:00:00', end_time='23:45:00'):
    '''
    Input : pred, type=array, size=(nb_time_series, nb_days*nb_time_step)
    Output : pd.DataFrame with colummns = index + time_series
    
    
    '''
    datetime_array = np.array([j for i in [build_timestamp_list(d+' '+start_time, d+' '+end_time, time_step_second=time_step_second) for d in days] for j in i])
    
    data = np.concatenate([datetime_array.reshape(1, datetime_array.shape[0]), pred]).T
    
    df = pd.DataFrame(data = data, columns = [index]+time_series).set_index(index).astype(float).round(3).reset_index()
    
    return df


