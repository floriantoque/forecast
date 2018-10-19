"""
        Load configuration of the model from yaml file
        Example:
        # python rf_optimize.py --config config_optimize.yaml

        #python rf_optimize.py --config config_optimize.yaml --force True;rf_fit.py --config config_fit.yaml --force True;rf_predict.py --config config_predict.yaml --force True;
"""

import yaml
import argparse

import os
import sys

sys.path.insert(0, '../../')
from utils.utils import *
from rf_model import *

import pandas as pd


parser = argparse.ArgumentParser(description='Parameters of script optimize')
parser.add_argument('--config', type=str, help='Yaml file containing the optimization configuration')
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
features_list = config['features_list']
time_series = config['time_series']
model_name = config['model_name']
start_date = config['start_date']
end_date = config['end_date']
path_to_save = config['path_to_save']
param_kfold = config["param_kfold"]
param_grid = config["param_grid"]
scoring = config["scoring"]
scaler_choice = config["scaler_choice"]


assert ((observation_data_path != None) & (exogenous_data_path != None) & (features_list != None) &
           (time_series != None) & (model_name != None) & (start_date != None) & (end_date != None)  &
           (path_to_save != None) & (param_kfold != None) & (param_grid != None) & (scoring != None) &
           (scaler_choice != None) ),'Configuration not well defined'

path_directory_to_save = path_to_save + model_name + '/'

print("You are going to optimize and create the model: {}".format(model_name))
create_model = True
if os.path.exists(path_directory_to_save) & (not (args.force)):
    create_model = yes_or_no(
        "WARNING !!!!!!!\n"
        "The model {} and his optimization saved in path: {} already exists.\n"
        "Do you want to erase and replace his optimization?".format(model_name, path_directory_to_save ))


if create_model:

    print('Read data: observation and exogenous')

    all_f = list(set([e for i in features_list for e in i]))
    df_Xy = read_csv_list(observation_data_path).set_index('Datetime').join(read_csv_list(exogenous_data_path).set_index('Datetime'))[start_date:end_date].dropna()

    X_list = [df_Xy[features].values for features in features_list]
    y = df_Xy[time_series].values

    my_model = Rf_model(model_name, start_date, end_date, features_list, time_series, observation_data_path,
                        exogenous_data_path, scaler_choice)

    print("Optimizing the model..")
    grid_search_dict = my_model.optimize(X_list, y, param_grid, param_kfold, scoring)
    print("Optimization done")

    print("Saving optimization result..")
    save_pickle(path_directory_to_save+"grid_search_dict.pkl",grid_search_dict)
    print("Saving optimization result done")
    
    print("Saving model..")
    my_model.save(path_directory_to_save, os.getcwd() + "/" + config_file)
    print("Saving model done")

else:
    sys.exit('Optimization aborted')

