import sys
import os

sys.path.insert(0, '../../')

from utils.utils_date import *
from utils.utils import *
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

observation_data_path = config['observation_data_path']
exogenous_data_path = config['exogenous_data_path']
start_date = config['start_date']
end_date = config['end_date']
path_to_save_prediction = config['path_to_save_prediction']
model_path = config['model_path']

assert (exogenous_data_path != None) &  (model_path != None) & (start_date != None) & (end_date != None) & (path_to_save_prediction != None),\
       'Configuration is not well defined, every variable have to be different of None'



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
        list_date = build_date_list(start_date,end_date)
    else:
        list_date = read_csv_list(observation_data_path)[["Datetime"]].set_index("Datetime")[start_date:end_date].\
            reset_index()["Datetime"].values.tolist()


    X = read_csv_list(exogenous_data_path).set_index('Datetime').ix[list_date].dropna()[my_model.infos['features']]
    list_date = X.index.values.tolist()

    pred_mean = my_model.predict(X.values, choice='mean')
    pred_median = my_model.predict(X.values, choice='median')

    print ("Prediction done")

    data_mean = [[i] + list(j) for i, j in zip(list_date, list(pred_mean))]
    data_median = [[i] + list(j) for i, j in zip(list_date, list(pred_median))]

    df_res_mean = pd.DataFrame(data=data_mean, columns=['Datetime']+my_model.infos['time_series']).round(2)
    df_res_median = pd.DataFrame(data=data_median, columns=['Datetime']+my_model.infos['time_series']).round(2)


    #Save the prediction
    print ("Saving the prediction...")

    if not os.path.exists(path_directory_to_save):
        os.makedirs(path_directory_to_save)

    df_res_mean.to_csv(path_directory_to_save+ start_date.split(" ")[0] +"_"+ end_date.split(" ")[0] + "_mean.csv",index=False)
    df_res_median.to_csv(path_directory_to_save + start_date.split(" ")[0] + "_" + end_date.split(" ")[0] + "_median.csv", index=False)
    print ("Saving the prediction done")

