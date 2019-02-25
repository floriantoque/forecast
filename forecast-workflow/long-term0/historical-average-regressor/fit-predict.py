import sys

sys.path.insert(0,  '../../../')

import model.historical_average.ha_model as ha_model
import utils.utils_regressor as utils_regressor
import utils.utils_date as utils_date
import utils.utils as utils
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import yaml
import argparse
import numpy as np

# Read config file and load config variables
parser = argparse.ArgumentParser(description='Parameters of script fit')
parser.add_argument('--config', type=str, help='Yaml file containing the configuration of the model')
parser.add_argument('--config_path', type=str, help='Yaml file containing the configuration path', default='../../config/config-path/default-config-path.yaml')
parser.add_argument('--force', type=bool,
                    help='Boolean: If True force all the saving even if file already exist, if False ask to the user',
                    default=False)
parser.add_argument('--n_jobs', type=int,
		    help='Int: Number of cores used during the optimization',
                    default=-1)

args = parser.parse_args()
config_file = args.config
config_path_file = args.config_path

with open(config_file, 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

with open(config_path_file, 'r') as stream:
    try:
        config_path = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

force_save = args.force
#n_jobs = args.n_jobs

features_time_step = config['features_time_step']
features_day = config['features_day']
features_todummy = config['features_todummy']
time_series = config['time_series']
start_datetime = config['start_datetime']
end_datetime = config['end_datetime']
scaler_choice_X = config['scaler_choice_X']
scaler_choice_y = config['scaler_choice_y']
start_datetime_optimization = config['start_datetime_optimization']
end_datetime_optimization = config['end_datetime_optimization']
#param_kfold = config['param_kfold']
#param_grid = config['param_grid']
best_params_ = config['best_params_']
best_params_time_series = config['best_params_time_series']
index = config['index']
start_time = config['start_time']
end_time = config['end_time']
time_step_second = config['time_step_second']
model_name = config['model_name']
estimator_choice = config['estimator_choice']
path_save = config_path['path_save']
observation_data_path = config_path['observation_data_path']
features_data_path = config_path['features_data_path']




# Load data
path_optimization = path_save+'optimize/{}/'.format(model_name)
path_prediction = path_save+'prediction/{}/'.format(model_name)
path_fit = path_save+'fit/{}/'.format(model_name)

model_infos = {}
model_infos['model_name'] = model_name
model_infos['estimator_choice'] = estimator_choice
model_infos['index'] = index
model_infos['start_time'] = start_time
model_infos['end_time'] = end_time
model_infos['time_step_second'] = time_step_second
model_infos['path_save'] = path_save
model_infos['path_optimization'] = path_optimization
model_infos['path_fit'] = path_fit
model_infos['path_prediction'] = path_prediction
model_infos['observation_data_path'] = observation_data_path
model_infos['features_data_path'] = features_data_path
#model_infos['param_grid'] = param_grid
#model_infos['param_kfold'] = param_kfold
model_infos['best_params_'] = best_params_
model_infos['best_params_time_series'] = best_params_time_series
model_infos['features_day'] = features_day
model_infos['features_todummy'] = features_todummy
model_infos['time_series'] = time_series
model_infos['start_datetime'] = start_datetime
model_infos['end_datetime'] = end_datetime
model_infos['scaler_choice_X'] = scaler_choice_X
model_infos['scaler_choice_y'] = scaler_choice_y
model_infos['start_datetime_optimization'] = start_datetime_optimization
model_infos['end_datetime_optimization'] = end_datetime_optimization
#model_infos['n_jobs'] = n_jobs


df_obs = utils.read_csv_list(observation_data_path).set_index(index).loc[start_datetime:end_datetime].reset_index()
df_fea = utils.read_csv_list(features_data_path).set_index(index).loc[start_datetime:end_datetime].reset_index()



# Data shaping
dfXy, features_time_step_name = utils.create_dfXy_long_term0(df_obs, df_fea, time_series,
                                                             features_time_step, features_day,
                                                             features_todummy=features_todummy,
                                                             index=index, start_time=start_time,
                                                             end_time=end_time, time_step_second=time_step_second)

X, y_list, X_names, days = utils.create_xy_dataset_long_term0(dfXy, time_series,
                                                             features_time_step_name=features_time_step_name,
                                                             index=index, start_time=start_time, end_time=end_time,
                                                             reduce=True)

#dfX = pd.DataFrame(data = np.concatenate([np.array(days).reshape(1,np.array(days).shape[0]),X.T]).T,
#                   columns=['Datetime'] + X_names.tolist())



# Split train - (test)
df_train_datetime = dfXy[[index]].copy()
df_train_datetime[index] = [i[:10]+' '+start_time for i in df_train_datetime[index]]
df_train_datetime = df_train_datetime.drop_duplicates()
index_train = len(df_train_datetime.set_index(index).loc[start_datetime_optimization:end_datetime_optimization])
X_train, y_list_train, days_train = X[:index_train], y_list[:,:index_train], days[:index_train]
#X_test, y_list_test, days_test = X[index_train:], y_list[:,index_train:], days[index_train:]



# Scaler
# TODO
n = 68


# Fit

# Create manually the best_params
if (best_params_ != None):
    best_params_time_series = {i: {'best_params_':best_params_} for i in np.arange(len(time_series[:n]))}
estimator_time_series = utils_regressor.fit_multioutput_regressor_multiseries_model(X_train,
                                                                                    y_list_train[:n],
                                                                                    estimator_choice,
                                                                                    best_params_time_series)
# Predict
pred = utils_regressor.predict_multioutput_regressor_multiseries_model(X, estimator_time_series)
df = utils.pred_day_array_to_df(pred, time_series[:n], days,
                                time_step_second=time_step_second, index=index)

utils.df_to_csv_safe(path_prediction + start_datetime[:10] + '_' + end_datetime[:10] + '.csv',
                     df, force_save=force_save)

