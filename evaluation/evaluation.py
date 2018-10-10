"""
Example
python evaluation.py --config config_evaluation.yaml
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0,'../')
from utils.utils import *

import yaml
import argparse


"""
Functions
"""

AT = 5

def get_list_common_date(start, end, obs, list_df_pred):
    df = pd.concat([obs[['Datetime']].set_index('Datetime')[start:end]]+
                   [df_[['Datetime']].set_index('Datetime') for df_ in list_df_pred], axis=1, join='inner')
    return df.index.values.astype(str)


def mse(obs, pred):
    return ((pred - obs) ** 2).mean()


def rmse(obs, pred):
    return np.sqrt(mse(obs, pred))


def mae(obs, pred):
    return np.absolute(pred - obs).mean()


def mape_at(obs, pred):
    mask = obs >= AT
    return ((np.absolute(pred[mask] - obs[mask]) / obs[mask]).mean())*100


def get_errors(obs, list_df_pred, list_name_pred, list_date, errors=[rmse, mae, mape_at, mse],
               errors_name=['rmse', 'mae', 'mape_at' + str(AT), 'mse'], ):
    data = []
    columns = ['model'] + errors_name
    obs = obs.set_index('Datetime').loc[list_date].values
    for df, name in zip(list_df_pred, list_name_pred):
        pred = df.set_index('Datetime').loc[list_date].values
        data.append([name] + [e(obs, pred).mean() for e in errors])
    return pd.DataFrame(data, columns=columns)


def error_per_station(obs, list_df_pred, list_name_pred, list_date, error=mape_at):
    data = []
    columns = ['__time_series__'] + list(obs.columns.values[1:])
    obs = obs.set_index('Datetime').loc[list_date].values
    for df, name in zip(list_df_pred, list_name_pred):
        pred = df.set_index('Datetime').loc[list_date].values
        data.append([name] + [error(obs[:, i], pred[:, i]) for i in range(pred.shape[1])])
    df = pd.DataFrame(data, columns=columns).set_index('__time_series__').T.reset_index()
    df.columns = np.array(['__time_series__'] + df.columns[1:].values.tolist()).astype(str)
    df['__index__'] = np.arange(len(df))
    return df


"""
main
"""

print('Read config file..')

parser = argparse.ArgumentParser(description='Parameters of script predict')
parser.add_argument('--config', type=str, help='Yaml file containing the configuration of the evaluation')
parser.add_argument('--force', type=bool, help='Boolean: If True force all the saving, if False ask to the user',
                    default=False)
args = parser.parse_args()
config_file = args.config

with open(config_file, 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

AT = config['AT']
start_date_train = config['start_date_train']
end_date_train = config['end_date_train']
start_date_test = config['start_date_test']
end_date_test = config['end_date_test']
file_path_obs = config['file_path_obs']
file_path_pred = config['file_path_pred']
list_name_pred = config['list_name_pred']
time_series = config['time_series']
time_series_name = config['time_series_name']
if time_series_name != None:
    try:
        time_series_name = [i.decode('utf-8') for i in time_series_name]
    except:
        time_series_name = config['time_series_name']
directory_path_to_save = config['directory_path_to_save']
name_evaluation = config['name_evaluation']


assert (start_date_train != None) & (end_date_train != None) & (start_date_test != None) & (end_date_test != None) &\
       (file_path_obs != None) & (file_path_pred != None) & (list_name_pred != None) & (time_series != None) &\
       (directory_path_to_save != None) & (name_evaluation != None),\
    'Configuration is not well defined, every variable have to be different of None'


print('Read observation and prediction files..')

obs = read_csv_list(file_path_obs)[['Datetime'] + time_series]
list_df_pred = [read_csv_list([f])[['Datetime'] + time_series] for f in file_path_pred]


print('Evaluate (1/2) global errors..')

for df in list_df_pred:
    assert ((df.columns != obs.columns.values).sum() == 0)

list_date_train = get_list_common_date(start_date_train, end_date_train, obs, list_df_pred)
list_date_test = get_list_common_date(start_date_test, end_date_test, obs, list_df_pred)
print(sorted(list(set([i[:10] for i in list_date_test]))))


df_train_errors = get_errors(obs, list_df_pred, list_name_pred, list_date_train,
                             errors_name=['rmse', 'mae', 'mape_at' + str(AT), 'mse'])
df_test_errors = get_errors(obs, list_df_pred, list_name_pred, list_date_test,
                            errors_name=['rmse', 'mae', 'mape_at' + str(AT), 'mse'])

if not args.force:
    if not os.path.exists(directory_path_to_save + name_evaluation + '/'):
        os.makedirs(os.path.dirname(directory_path_to_save + name_evaluation + '/'))
    elif not yes_or_no("Folder " + name_evaluation + " already exists. Do you want to continue?"):
        sys.exit()


df_train_errors.round(2).to_csv(directory_path_to_save + name_evaluation + "/errors_trainset.csv", index=False)
df_test_errors.round(2).to_csv(directory_path_to_save + name_evaluation + "/errors_testset.csv", index=False)


print('Evaluate (2/2) errors per time-series..')
palette = sns.color_palette("hls",n_colors=20)
df_train_error_per_station = error_per_station(obs, list_df_pred, list_name_pred, list_date_train, error=mape_at)
df_test_error_per_station = error_per_station(obs, list_df_pred, list_name_pred, list_date_test, error=mape_at)

if time_series_name != None:
    dict_id_name = dict(zip(time_series, time_series_name))
    df_train_error_per_station['__time_series__'] = [dict_id_name[i] for i in
                                                     df_train_error_per_station['__time_series__'].values]
    df_test_error_per_station['__time_series__'] = [dict_id_name[i] for i in
                                                    df_test_error_per_station['__time_series__'].values]


sns.set_style('whitegrid')
ax = plt.gca()
kind = 'scatter'

f = df_train_error_per_station.plot(figsize=(20, 4), kind=kind, x='__index__', y=list_name_pred[0], ax=ax, color=palette[0])
for i, m in enumerate(list_name_pred[1:]):
    f = df_train_error_per_station.plot(kind=kind, x='__index__', y=m, ax=ax, color=palette[i+1])
f.set_xticks(df_train_error_per_station['__index__'])
f.set_xticklabels(df_train_error_per_station['__time_series__'], rotation=90)
f.set_xlabel('Time-series')
f.set_ylabel('MAPE@' + str(AT))

ax.legend(list_name_pred)
fig = f.get_figure()
fig.savefig(directory_path_to_save + name_evaluation + "/errors_trainset_per_time_series.png", bbox_inches='tight')
plt.close()

sns.set_style('whitegrid')
ax = plt.gca()
kind = 'scatter'

f = df_test_error_per_station.plot(figsize=(20, 4), kind=kind, x='__index__', y=list_name_pred[0], ax=ax, color=palette[0])
for i, m in enumerate(list_name_pred[1:]):
    f = df_test_error_per_station.plot(kind=kind, x='__index__', y=m, ax=ax, color=palette[i+1])
f.set_xticks(df_test_error_per_station['__index__'])
f.set_xticklabels(df_test_error_per_station['__time_series__'], rotation=90)
f.set_xlabel('Time-series')
f.set_ylabel('MAPE@' + str(AT))

ax.legend(list_name_pred)
fig = f.get_figure()
fig.savefig(directory_path_to_save + name_evaluation + "/errors_testset_per_time_series.png", bbox_inches='tight')
plt.close()

print('Evaluation done')

