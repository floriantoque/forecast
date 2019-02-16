from sarimax_functions import *

import yaml
import argparse
import os


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

force = args.force
path_model_predict = config['path_model_predict']
path_model_infos = config['path_model_infos']
start_train = config['start_train']
end_train = config['end_train']
end_test = config['end_test']


model_infos = utils.load_pickle(path_model_infos)
model_infos['path_model_predict'] = path_model_predict

path_grid_search = model_infos['path_grid_search']
grid_search_params = utils.load_pickle(path_grid_search)

obs_path = model_infos['obs_path']
fea_path = model_infos['fea_path']
features_todummy = model_infos['features_todummy']
features_nottodummy = model_infos['features_nottodummy']
time_series = model_infos['time_series']


obs = utils.read_csv_list(obs_path)
fea = utils.read_csv_list(fea_path)

fea = utils.df_todummy_df(fea, features_todummy, features_nottodummy)
obs = obs.set_index('Datetime')[time_series].reset_index()


# Best params
configs = sarima_configs(model_infos['param_grid'])
best_params = {}

for ts in obs.set_index('Datetime').columns.values:
    res = [(str(cfg), np.array(grid_search_params[ts][str(cfg)]).mean()) for cfg in configs] 
    res.sort(key=lambda tup: tup[1])
    best_params[ts] = ast.literal_eval(res[0][0])
model_infos['best_params'] = best_params

# Forecasting
print('Forecasting...')
Yendog, exog, Yendog_name_list, exog_name_list = create_Xy_basic(start_train, end_test, obs, fea)

index_train = len(utils_date.get_list_common_date(start_train, end_train, obs, [fea]))
Yendog_train, Yendog_test = Yendog[:index_train], Yendog[index_train:]
exog_train, exog_test = exog[:index_train], exog[index_train:]

predictions = []
for idx,ts in enumerate(tqdm(time_series, desc='Predict (Time series loop)')):
    predictions.append(fit_and_predict(Yendog_train[:,idx].flatten(), exog_train, exog_test, best_params[ts]))

predictions = np.array(predictions).T    
predictions[predictions<0]=0


df = obs.set_index('Datetime').loc[end_train:end_test].reset_index().copy()
for idx,ts in enumerate(time_series):
    df[ts] = predictions[:,idx]
try:
    df.to_csv('{}{}/{}_{}.csv'.format(path_model_predict, model_infos['name'], end_train, end_test),index=False)
except FileNotFoundError:
    os.makedirs('{}{}/'.format(path_model_predict, model_infos['name']))
    df.to_csv('{}{}/{}_{}.csv'.format(path_model_predict, model_infos['name'], end_train, end_test),index=False)

    
utils.save_pickle_safe(path_model_infos, model_infos, force_save = force)  

