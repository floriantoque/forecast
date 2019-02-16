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



observation_path = config['observation_path']
context_path = config['context_path']
features = config['features']
features_day = config['features_day']
time_series = config['time_series']
model_name = config['model_name']
start_date = config['start_date']
end_date = config['end_date']
path_to_save = config['path_to_save']

assert (observation_path != None) & (context_path != None) & (features != None) & (features_day != None) & \
       (time_series != None) & (model_name != None) & (start_date != None) & (end_date != None) & (path_to_save != None),\
       'Configuration is not well defined, every variable have to be different of None'


print("You are going to create the model: %s" % model_name)
create_model = True
if os.path.exists(path_to_save+model_name+'/') & (not (args.force)):
    create_model = yes_or_no(
        "WARNING !!!!!!!\n"
        "The model {} saved in path: {} already exists.\n"
        "Do you want to erase and replace it?".format(model_name, path_to_save+model_name+'/'))

if create_model:
    print('Read data')

    df_observation = read_csv_list(observation_path)
    days = sorted(list(set([i[:10] for i in df_observation['Datetime'].values])))
    timestamp_list = [j for i in [build_timestamp_list(d+' 00:00:00', d+' 23:45:00', time_step_second=15*60) for d in days] for j in i]
    df_date = pd.DataFrame(data=timestamp_list, columns=['Datetime']).set_index('Datetime')
    df_observation = df_date.join(df_observation.set_index('Datetime')).fillna(0).reset_index()

    df_context = read_csv_list(context_path)


    my_model = Ha_inverted_model(model_name)

    my_model.infos['start_date'] = start_date
    my_model.infos['end_date'] = end_date
    my_model.infos['time_series'] = time_series
    my_model.infos['features'] = features
    my_model.infos['features_day'] = features_day
    my_model.infos['observation_path'] = observation_path
    my_model.infos['context_path'] = context_path
    my_model.infos['path_to_save'] = path_to_save

    dfXy = df_observation.set_index('Datetime')[start_date: end_date].join(df_context.set_index('Datetime')).reset_index()
    X, ylist, features_end = build_Xylist(dfXy, features, features_day, time_series, start_date, end_date)

    print('Fit model')
    my_model.fit(X, ylist)
    print('Fit model done')

    print('Save model')
    save_pickle(path_to_save+model_name+'/'+model_name+'.pkl', my_model)
    print('Save model done')

