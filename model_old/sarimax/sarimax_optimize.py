from sarimax_functions import *

import yaml
import argparse


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
obs_path = config['observation_data_path']
fea_path = config['features_data_path']
features_todummy = config['features_todummy']
features_nottodummy = config['features_nottodummy']
time_series = config['time_series']
model_name = config['model_name']
path_model_optimization = config['path_model_optimization']
param_grid = config['param_grid']
start_train_list = config['start_train_list']
end_train_list = config['end_train_list']
end_val_list = config['end_val_list']
n_jobs = config['n_jobs']
parallel = config['parallel']


obs = utils.read_csv_list(obs_path)
fea = utils.read_csv_list(fea_path)

fea = utils.df_todummy_df(fea, features_todummy, features_nottodummy)
obs = obs.set_index('Datetime')[time_series].reset_index()



path_grid_search = path_model_optimization+model_name+'/grid_search.pkl'
path_model_infos = path_model_optimization+model_name+'/model_infos.pkl'

model_infos = {}
model_infos['name'] = model_name
model_infos['path_model_optimization'] = path_model_optimization
model_infos['time_series'] = time_series
model_infos['features'] = fea.set_index('Datetime').columns.values
model_infos['obs_path'] = obs_path
model_infos['fea_path'] = fea_path
model_infos['param_grid'] = param_grid
model_infos['start_train_list'] = start_train_list
model_infos['end_train_list'] = end_train_list
model_infos['end_val_list'] = end_val_list
model_infos['path_grid_search'] = path_grid_search
model_infos['path_model_infos'] = path_model_infos
model_infos['features_todummy'] = features_todummy
model_infos['features_nottodummy'] = features_nottodummy

configs = sarima_configs(param_grid)



print('Optimization...')
cpt=1

grid_search_params = {}
for start_train, end_train, end_val in zip(start_train_list, end_train_list, end_val_list):
    print('Iteration {}/{}'.format(cpt,len(start_train_list)))
    cpt+=1
    Yendog, exog, Yendog_name_list, exog_name_list = create_Xy_basic(start_train, end_val, obs, fea)
    
    index_train = len(utils_date.get_list_common_date(start_train, end_train, obs, [fea]))
    Yendog_train = Yendog[:index_train]
    #mean_ = Yendog_train.mean()
    #std_ = Yendog_train.std()
    #Yendog_train =(Yendog[:index_train]-mean_)/std_
    Yendog_val = Yendog[index_train:]
    
    exog_train, exog_val = exog[:index_train], exog[index_train:]

    for ts_index, ts in enumerate(time_series):
        
        scores = grid_search(Yendog_train[:, ts_index], Yendog_val[:, ts_index],
                             exog_train, exog_val, configs, n_jobs=n_jobs, parallel=parallel, ts_index=ts_index+1)
        for c,s in scores:
            try:
                grid_search_params[ts]
            except:
                grid_search_params[ts]={}
            try:
                grid_search_params[ts][c].append(s)
            except:
                grid_search_params[ts][c]=[]
                grid_search_params[ts][c].append(s)

utils.save_pickle_safe(path_grid_search, grid_search_params, force_save = force)
utils.save_pickle_safe(path_model_infos, model_infos, force_save = force)  

