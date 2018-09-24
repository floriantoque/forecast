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


