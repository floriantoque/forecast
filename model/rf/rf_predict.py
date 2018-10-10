import sys
import os

sys.path.insert(0, '../../')
from utils.utils import *
from utils.utils_date import *
from rf_model import *

import yaml
import argparse

import pandas as pd

parser = argparse.ArgumentParser(description='Parameters of script predict')
parser.add_argument('--config', type=str, help='Yaml file containing the configuration of the prediction')
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
start_date = config['start_date']
end_date = config['end_date']
path_to_save_prediction = config['path_to_save_prediction']
model_path = config['model_path']
rf_object_learned_path = config['rf_object_learned_path']
time_step_second = config['time_step_second']

assert (exogenous_data_path != None) &  (model_path != None) & (start_date != None) & (end_date != None) &\
       (path_to_save_prediction != None) & (rf_object_learned_path != None),\
       'Configuration is not well defined, every variable have to be different of None'

if observation_data_path == None:
    assert(time_step_second != None)

print ("Loading the model...")
my_model = load_pickle(model_path)
print ("Loading the model done")

path_directory_to_save = path_to_save_prediction + my_model.infos['name'] + '/'




print("You are going to predict with the model: {} and save the prediction in folder {}".format(my_model.infos['name'], path_directory_to_save))

create_prediction = True
if os.path.exists(path_directory_to_save) & (not (args.force)):
    create_prediction = yes_or_no(
        "WARNING !!!!!!!\n"
        "The directory: {} already exists.\n"
        "Do you want to possibly erase and replace the prediction saved in this directory?".format(path_directory_to_save))



if create_prediction:

    print ("Prediction...")
    if observation_data_path == None:
        list_date = build_timestamp_list(start_date, end_date, time_step_second)
    else:
        list_date = read_csv_list(observation_data_path)[["Datetime"]].set_index("Datetime")[start_date:end_date].\
            reset_index()["Datetime"].values.tolist()

    rf = load_pickle(rf_object_learned_path)
    df_X = read_csv_list(exogenous_data_path).set_index('Datetime')[my_model.infos['features']].ix[list_date]

    for d in df_X[df_X.isnull().any(axis=1)].index.values:
        print('Datetime {} will not be predicted because it is not in exogenous data or observation data'.format(d))

    df_X = df_X.dropna()
    list_date = df_X.index.values.tolist()

    X = df_X.values

    pred = my_model.predict(rf, X)

    data = [[i] + list(j) for i, j in zip(list_date, list(pred))]
    df_res = pd.DataFrame(data=data, columns=['Datetime'] + my_model.infos['time_series']).round(2)
    print ("Prediction done")

    #Save the prediction
    print ("Saving the prediction...")

    if not os.path.exists(path_directory_to_save):
        os.makedirs(path_directory_to_save)

    df_res.to_csv(path_directory_to_save + start_date.split(" ")[0] + "_" + end_date.split(" ")[0] + '.csv', index=False)

    print ("Saving the prediction done")

