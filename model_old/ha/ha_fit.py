"""
        Load configuration of the model from yaml file
        Example:
        # python ha_fit.py --config config_fit.yaml
"""

import yaml
import argparse

import os
import sys

sys.path.insert(0, '../../')

from utils.utils import *
from ha_model import *

import pandas as pd




parser = argparse.ArgumentParser(description='Parameters of script fit')
parser.add_argument('--config', type=str, help='Yaml file containing the configuration of the model')
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
features = config['features']
time_series = config['time_series']
model_name = config['model_name']
start_date = config['start_date']
end_date = config['end_date']
path_to_save = config['path_to_save']

assert (observation_data_path != None) & (exogenous_data_path != None) & (features != None) & \
       (time_series != None) & (model_name != None) & (start_date != None) & (end_date != None) & (path_to_save != None),\
       'Configuration is not well defined, every variable have to be different of None'

path_directory_to_save = path_to_save + model_name + '/'



print("You are going to create the model: %s" % model_name)
create_model = True
if os.path.exists(path_directory_to_save) & (not (args.force)):
    create_model = yes_or_no(
        "WARNING !!!!!!!\n"
        "The model {} saved in path: {} already exists.\n"
        "Do you want to erase and replace it?".format(model_name, path_directory_to_save))

if create_model:
    my_model = Ha_model(model_name, start_date, end_date, features, time_series, observation_data_path, exogenous_data_path )

    print('Read data: observation and exogenous')

    df_observation = read_csv_list(observation_data_path)
    df_exogenous = read_csv_list(exogenous_data_path)

    df_Xy = df_observation.set_index('Datetime')[time_series].join(df_exogenous.set_index("Datetime")).dropna()[start_date:end_date]

    X = df_Xy[features].values
    y = df_Xy[time_series].values

    print("Learning the model..")
    my_model.fit(X,y)
    print("Learning done")

    print("Saving model..")
    my_model.save(path_directory_to_save, os.getcwd() + "/" + config_file)
    print("Saving model done")




