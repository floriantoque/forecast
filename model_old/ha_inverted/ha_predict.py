import sys
import os

sys.path.insert(0, '../../')
from utils.utils import *
from utils.utils_date import *

from ha_model import *

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

observation_path = config['observation_path']
context_path = config['context_path']
start_date = config['start_date']
end_date = config['end_date']
path_save_prediction = config['path_save_prediction']
model_path = config['model_path']

assert (context_path != None) & (model_path != None) & (start_date != None) & (end_date != None) & (path_save_prediction != None),\
       'Configuration is not well defined, every variable have to be different of None'



print ("Loading the model...")
my_model = load_pickle(model_path)
print ("Loading the model done")

path_directory_to_save = path_save_prediction + my_model.infos['name'] + '/'




print("You are going to predict with the model: {} and save the prediction in folder {}".format(my_model.infos['name'], path_directory_to_save))

create_prediction = True
if os.path.exists(path_directory_to_save) & (not (args.force)):
    create_prediction = yes_or_no(
        "WARNING !!!!!!!\n"
        "The directory: {} already exists.\n"
        "Do you want to possibly erase and replace the prediction saved in this directory?".format(path_directory_to_save))



if create_prediction:

    print ("Prediction...")
    df_observation = read_csv_list(observation_path)
    days = sorted(list(set([i[:10] for i in df_observation['Datetime'].values])))
    timestamp_list = [j for i in
                      [build_timestamp_list(d + ' 00:00:00', d + ' 23:45:00', time_step_second=15 * 60) for d in days]
                      for j in i]
    df_date = pd.DataFrame(data=timestamp_list, columns=['Datetime']).set_index('Datetime')
    df_observation = df_date.join(df_observation.set_index('Datetime')).fillna(0).reset_index()

    df_context = read_csv_list(context_path)


    dfXy = df_observation.set_index('Datetime')[start_date: end_date].join(
        df_context.set_index('Datetime')).reset_index()
    datetime_list = dfXy['Datetime'].values

    X, features_end = build_X(dfXy, my_model.infos['features'], my_model.infos['features_day'], my_model.infos['time_series'], start_date, end_date)

    pred_all = my_model.predict(X)

    pred_all = np.swapaxes(pred_all, 0, 1)
    pred_all = pred_all.reshape(len(my_model.infos['time_series']), pred_all.shape[1] * pred_all.shape[2])
    df_pred = pd.DataFrame(data=datetime_list, columns=['Datetime'])
    for ix, ts in enumerate(my_model.infos['time_series']):
        df_pred[ts] = pred_all[ix]

    df_pred = df_pred.round(3)


    print ("Saving the prediction...")

    if not os.path.exists(path_save_prediction + my_model.infos['name'] + "/"):
        os.makedirs(path_save_prediction + my_model.infos['name'] + "/")

    df_pred.to_csv(path_save_prediction + my_model.infos['name'] + '/' + start_date[:10] + '_' + end_date[:10] + '.csv',
                   index=False)
    print ("Saving the prediction done")

