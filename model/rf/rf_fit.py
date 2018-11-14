"""
        Load configuration of the model from yaml file
        Example:
        # python rf_fit.py --config config_fit.yaml
"""

import yaml
import argparse

from rf_model import *

import pandas as pd

import os
import sys
sys.path.insert(0, '../../')
from utils.utils import *




parser = argparse.ArgumentParser(description='Parameters of script fit')
parser.add_argument('--config', type=str, help='Yaml file containing the learning configuration of the model')
parser.add_argument('--force', type=bool, help='Boolean: If True force all the saving, if False ask to the user',default=False)
args = parser.parse_args()
config_file = args.config


with open(config_file, 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

observation_data_path = config['observation_data_path']
exogenous_data_path = config['exogenous_data_path']
time_series = config['time_series']
model_name = config['model_name']
start_date = config['start_date']
end_date = config['end_date']
path_to_save = config['path_to_save']
path_grid_search_dict = config['path_grid_search_dict']
model_path = config['model_path']
features = config['features']
best_params = config['best_params']
scaler_choice = config['scaler_choice']
tminus = config['tminus']
tplus = config['tplus']
features_window = config['features_window']

assert( path_to_save != None), 'path_to_save has to be defined'

if (path_grid_search_dict != None) & (model_path != None):
    assert ((observation_data_path == None) & (exogenous_data_path == None) & (time_series == None) & (model_name == None) & \
           (start_date == None) & (end_date == None) & (features == None) & (best_params == None) & (scaler_choice == None)),\
            'Configuration is not well defined'
elif (observation_data_path != None) & (exogenous_data_path != None) & (time_series != None) & (model_name != None) & \
        (start_date != None) & (end_date != None) & (features != None) & (best_params != None) & \
        (scaler_choice != None) & (scaler_choice != None) & (tminus != None) & (tplus != None) & (features_window != None):
    assert (path_grid_search_dict == None) & (model_path == None), 'Configuration is not well defined'

if tminus != None:
    assert (tplus != None) & (features_window != None)
else:
    assert (tplus == None) & (features_window == None)

# Load/create model

if (path_grid_search_dict != None) & (model_path != None):

    infos = load_pickle(model_path)
    model_name = infos['name']
    start_date = infos['start_date']
    end_date = infos['end_date']
    features_list = infos['features_list']
    time_series = infos['time_series']
    observation_data_path = infos['observation_data_path']
    exogenous_data_path = infos['exogenous_data_path']
    scaler_choice = infos['scaler_choice']

    tminus = infos['tminus']
    tplus = infos['tplus']
    features_window = infos['features_window']

    my_model = Rf_model(model_name, start_date, end_date, features_list, time_series, observation_data_path,
                        exogenous_data_path, scaler_choice)

    my_model.infos['tminus'] = tminus
    my_model.infos['tplus'] = tplus
    my_model.infos['features_window'] = features_window

    print("You are going to create/fit the model: {}".format(my_model.infos['name']))

    path_directory_to_save = path_to_save + my_model.infos['name'] + '/'
    fit_model = True
    if os.path.exists(path_directory_to_save) & (not (args.force)):
        fit_model = yes_or_no(
            "WARNING !!!!!!!\n"
            "The model {} saved in path: {} already exists.\n"
            "Do you want to erase and replace it?".format(my_model.infos['name'], path_directory_to_save))

    if fit_model:
        # Selection of the best features and best params with grid search list
        grid_search_dict = load_pickle(path_grid_search_dict)
        best_conf = [(features, grid_search_dict[tuple(features)].best_params_,
                      grid_search_dict[tuple(features)].best_score_) for features in list(grid_search_dict.keys())]
        best_conf.sort(key=lambda x: x[2])
        features = list(best_conf[-1][0])
        best_params = best_conf[-1][1]
        my_model.infos['features'] = features
        my_model.infos['best_params'] = best_params

        print('Read data: observation and exogenous')
        df_Xy = read_csv_list(my_model.infos['observation_data_path']).set_index("Datetime").join(
            read_csv_list(my_model.infos['exogenous_data_path']).set_index("Datetime"))[
            my_model.infos['start_date']:my_model.infos['end_date']].dropna()

        if features_window!=None:
            X, y, fname = window_Xy(df_Xy, time_series, features, features_window, tminus, tplus)
        else:
            X = df_Xy[my_model.infos['features']].values
            y = df_Xy[my_model.infos['time_series']].values

    else:
        sys.exit("Answer = No => Exit script fit")

else:

    print("You are going to create/fit the model: {}".format(model_name))

    path_directory_to_save = path_to_save + model_name + '/'
    fit_model = True
    if os.path.exists(path_directory_to_save) & (not (args.force)):
        fit_model = yes_or_no(
            "WARNING !!!!!!!\n"
            "The model {} saved in path: {} already exists.\n"
            "Do you want to erase and replace it?".format(model_name, path_directory_to_save))

    if fit_model:
        print('Read data: observation and exogenous')
        df_Xy = read_csv_list(observation_data_path).set_index("Datetime").join(
            read_csv_list(exogenous_data_path).set_index("Datetime"))[start_date:end_date].dropna()

        if features_window!=None:
            X, y, fname = window_Xy(df_Xy, time_series, features, features_window, tminus, tplus)
        else:
            X = df_Xy[features].values
            y = df_Xy[time_series].values


        # X = df_Xy[features].values
        # y = df_Xy[time_series].values

        my_model = Rf_model(model_name, start_date, end_date, [features], time_series, observation_data_path,
                            exogenous_data_path, X, scaler_choice)

        my_model.infos['tminus'] = tminus
        my_model.infos['tplus'] = tplus
        my_model.infos['features_window'] = features_window
        my_model.infos['features'] = features
        my_model.infos['best_params'] = best_params

    else:
        sys.exit("Answer = No => Exit script fit")




# Learn and save model

print("Learning the model..")
rf = my_model.fit(X, y, my_model.infos['best_params'])
print("Learning done")

if my_model.infos['features_window'] == None:
    fname = features
feature_importances = dict(zip(fname, np.round(rf.feature_importances_*100, 2).tolist()))
my_model.infos["feature_importances"] = feature_importances

print("Saving model information..")
my_model.save(path_directory_to_save, os.getcwd() + "/" + config_file)
print("Saving model information done")


print("Saving model ..")
save = True
if not(args.force):
    size = size_of_object(rf)
    save = yes_or_no(
       "WARNING !!!!!!!\n"
       "The rf model will take approximately a disk space of {:.2f}Mo. Do you want to save it?".format(size/1000000.))

if save:
    save_pickle(path_directory_to_save+'rf_object_learned.pkl', rf)
    print("Saving model done")
else:
    print("Model not saved")


