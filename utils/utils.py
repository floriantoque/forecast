try:
    import cPickle as pickle
except:
    import pickle

import os
import numpy as np
import pandas as pd

from scipy import spatial

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