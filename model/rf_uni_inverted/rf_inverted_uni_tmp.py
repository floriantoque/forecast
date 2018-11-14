import sys
sys.path.insert(0, '../../utils/')
from utils import *
from pylab import *
from utils_date import *
import pickle
from tqdm import tqdm
import copy

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

import itertools
import pandas as pd



def create_xy_dataset(df_Xy, time_series, features_exogenous, features_context):
    df_Xy = copy.deepcopy(df_Xy[time_series+features_exogenous+features_context].dropna())
    days = sorted(list(set([i[:10] for i in df_Xy.index.values])))
    Xnames = [f+'-T'+str(ix)for f in features_exogenous for ix in np.arange(96)] + features_context
    
    X = []
    list_y=[]
    for d in tqdm(days,desc='Days loop'):
        ex = df_Xy.loc[d+' 00:00:00': d+ ' 23:45:00'][features_exogenous].values.T.flatten()
        co = df_Xy.loc[[d+' 00:00:00']][features_context].values.flatten()
        X.append(np.concatenate([ex, co]))
        y = []
        for s in time_series:
            y.append(df_Xy.loc[d+' 00:00:00': d+ ' 23:45:00'][s].values)
        list_y.append(y)
        
    return np.array(X), np.swapaxes(np.array(list_y),0,1), Xnames, days
        

def fit_predict(X, y, cv, param_grid, scaler_choice_X, scaler_choice_y):
    
    pred_train_array = []
    pred_val_array = []
    ytrain_array = []
    yval_array = []
    
    for ix_train,ix_val in cv.split(X):
        Xtrain,Xval = X[ix_train], X[ix_val]
        ytrain,yval = y[ix_train], y[ix_val]

        scalerX = None
        scalery = None
        if scaler_choice_X == 'minmax':
            scalerX = MinMaxScaler(feature_range=(0, 1))
        elif scaler_choice_X == 'standard':
            scalerX = StandardScaler()
        if scalerX != None:
            Xtrain = scalerX.fit_transform(Xtrain)
            Xval = scalerX.transform(Xval)

        if scaler_choice_y == 'minmax':
            scalery = MinMaxScaler(feature_range=(0, 1))
        elif scaler_choice_y == 'standard':
            scalery = StandardScaler()
        if scalery != None:
            ytrain = scalery.fit_transform(ytrain)
            yval = scalery.transform(yval)

        keys, values = zip(*param_grid.items())
        all_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
        pred_train_params = []
        pred_val_params = []
        for p in all_params:
            rf = RandomForestRegressor(**p, verbose=0)
            rf.fit(Xtrain,ytrain)
            pred_train = rf.predict(Xtrain)
            pred_val = rf.predict(Xval)
            pred_train_params.append(pred_train)
            pred_val_params.append(pred_val)

        pred_train_array.append(pred_train_params)
        pred_val_array.append(pred_val_params)
        ytrain_array.append(ytrain)
        yval_array.append(yval)

    pred_train_array = np.array(pred_train_array)
    pred_val_array = np.array(pred_val_array)
    ytrain_array = np.array(ytrain_array)
    yval_array = np.array(yval_array) 
    
    return pred_train_array, pred_val_array, ytrain_array, yval_array


def optimize(X, y_list, param_kfold, time_series):

    cv = KFold(**param_kfold)
    
    # loop over y_list
    pred_train_all = []
    pred_val_all = []
    ytrain_all = []
    yval_all = []

    for y, ts in tqdm(zip(y_list,time_series),desc='Optimization time_series'):
        pred_train_array, pred_val_array, ytrain_array, yval_array = fit_predict(X, y, cv, param_grid, scaler_choice_X,
                                                                                 scaler_choice_y)
        pred_train_all.append(pred_train_array)
        pred_val_all.append(pred_val_array)
        ytrain_all.append(ytrain_array)
        yval_all.append(yval_array)

    pred_train_all = np.array(pred_train_all)
    pred_val_all = np.array(pred_val_all)

    ytrain_all = np.array(ytrain_all)
    yval_all = np.array(yval_all)

    # Errors calculus
    pred_train = np.swapaxes(pred_train_all, 0, 2)
    obs_train = np.swapaxes(ytrain_all, 0, 1)
    pred_val = np.swapaxes(pred_val_all, 0, 2)
    obs_val = np.swapaxes(yval_all, 0, 1)

    errors_function = [rmse, mse, mae, mape_at]
    errors_name = ['rmse', 'mse', 'mae', 'mape_at']
    grid_search_dict={'train':{}, 'val':{}}
    for ef,en in zip(errors_function, errors_name):

        grid_search_dict['train'][en]={}
        grid_search_dict['train'][en]['error'] = np.array([np.array([ef(np.concatenate(pred_train[ixp][ixcv]), np.concatenate(obs_train[ixcv])) 
                            for ixcv in range(pred_train.shape[1])]) for ixp in range(pred_train.shape[0])])

        grid_search_dict['train'][en]['mean'] = grid_search_dict['train'][en]['error'].mean(axis=1) 
        grid_search_dict['train'][en]['std'] = grid_search_dict['train'][en]['error'].std(axis=1) 

        grid_search_dict['val'][en]={}
        grid_search_dict['val'][en]['error'] = np.array([np.array([ef(np.concatenate(pred_val[ixp][ixcv]), np.concatenate(obs_val[ixcv])) 
                            for ixcv in range(pred_val.shape[1])]) for ixp in range(pred_val.shape[0])])

        grid_search_dict['val'][en]['mean'] = grid_search_dict['val'][en]['error'].mean(axis=1) 
        grid_search_dict['val'][en]['std'] = grid_search_dict['val'][en]['error'].std(axis=1) 

    return grid_search_dict


def pred_list_to_dataframe(pred_list, time_series, days):
    data = [j for i in [build_timestamp_list(d+' 00:00:00', d+ ' 23:45:00') for d in days] for j in i]
    df = pd.DataFrame(data=data, columns=['Datetime'])
    for ix, ts in enumerate(time_series):
        df[ts] = pred_list[ix].reshape(pred_list[ix].shape[0]*pred_list[ix].shape[1])
    return df



observation_data_path = ['/home/toque/data2/montreal/stm/data/valid_metro_15min_2015_2016_2017_sumpass_nodayfree.csv']
exogenous_data_path = ['/home/toque/data2/montreal/events/data/clean/events_2015_2018_end_event_stopid.csv',
                       '/home/toque/data2/montreal/events/data/clean/events_2015_2018_start_event_stopid.csv',
                       '/home/toque/data2/montreal/events/data/clean/events_2015_2018_period_event_stopid.csv',
                       '/home/toque/data2/weather/predicted_weather/predicted_weather_2015_2017_included_perday_pm.csv'
                      ]
context_data_path = ['/home/toque/data2/date/2013-01-01-2019-01-01_new.csv']

df_observation = read_csv_list(observation_data_path)
df_exogenous = read_csv_list(exogenous_data_path)
df_context = read_csv_list(context_data_path)

#Detrend Data to delete
pred_lt_rf_uni_inverted = pd.read_csv('/home/toque/data2/forecast/model/rf_uni_inverted/prediction/lt_rf_uni_inverted/2015-01-01_2017-12-31.csv')
df_observation = (df_observation.set_index('Datetime') - df_observation[['Datetime']].set_index('Datetime').join(pred_lt_rf_uni_inverted.set_index('Datetime'))).reset_index()

# fill timestamps not available with 0 to have 96 timestamps per day
days = sorted(list(set([i[:10] for i in df_observation['Datetime'].values])))
timestamp_list = [j for i in [build_timestamp_list(d+' 00:00:00', d+' 23:45:00', time_step_second=15*60) for d in days] for j in i]
df_date = pd.DataFrame(data = timestamp_list, columns = ['Datetime']).set_index('Datetime')
df_observation = df_date.join(df_observation.set_index('Datetime')).fillna(0).reset_index()



time_series = ['11', '32', '34', '15', '44', '65', '31', '33', '35', '47', '13',
       '14', '1', '9', '5', '18', '36', '24', '68', '43', '8', '64', '10',
       '55', '3', '49', '51', '2', '19', '56', '7', '6', '4', '48', '66',
       '25', '23', '28', '39', '54', '60', '27', '20', '46', '12', '21',
       '62', '52', '41', '50', '30', '16', '37', '40', '26', '67', '57',
       '61', '42', '45', '38', '29', '58', '63', '22', '59', '53', '17']

features_exogenous = ['5-end_event', '11-end_event', '12-end_event',
       '13-end_event', '15-end_event', '16-end_event', '23-end_event',
       '24-end_event', '31-end_event', '32-end_event', '35-end_event',
       '43-end_event', '45-end_event', '61-end_event', '68-end_event',
       '5-start_event', '11-start_event', '12-start_event', '13-start_event',
       '15-start_event', '16-start_event', '23-start_event', '24-start_event',
       '31-start_event', '32-start_event', '35-start_event', '43-start_event',
       '45-start_event', '61-start_event', '68-start_event', '5-period_event',
       '11-period_event', '12-period_event', '13-period_event',
       '15-period_event', '16-period_event', '23-period_event',
       '24-period_event', '31-period_event', '32-period_event',
       '35-period_event', '43-period_event', '45-period_event',
       '61-period_event', '68-period_event']

features_context = ['Day_id', 'Mois_id','vac_noel_quebec', 'day_off_quebec', '24DEC', '31DEC',
                    'renov_beaubien', 'vac_udem1', 'vac_udem2']

scaler_choice_X = None
scaler_choice_y = None

param_kfold={
    'n_splits': 5,
    'shuffle': True,
    'random_state': 1}

param_grid={
    'n_estimators': [100, 150, 200],
    'max_features': ['auto',None],
    'max_depth': [None],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,5,10],
    'n_jobs': [6],
    'criterion': ['mse']}


start_datetime, end_datetime = '2015-01-01 00:00:00', '2016-12-31 23:45:00'


model_name = 'mt_rf_uni_inverted_events15_detrended_lt_rf_uni_inverted'



# # Optimisation

df_Xy = df_observation.set_index('Datetime').join([df_context.set_index('Datetime'), df_exogenous.set_index('Datetime')])

df_Xy_train = df_Xy[start_datetime:end_datetime]
Xtrain, ytrain_list, Xnames, days = create_xy_dataset(df_Xy_train, time_series, features_exogenous, features_context)

grid_search_dict = optimize(Xtrain, ytrain_list, param_kfold, time_series)

save_pickle('/home/toque/data2/forecast/model/rf_uni_inverted/optimize/'+model_name+'/grid_search_dict.pkl', grid_search_dict)



# # Get best params and learn with best params

grid_search_dict = load_pickle('/home/toque/data2/forecast/model/rf_uni_inverted/optimize/'+model_name+'/grid_search_dict.pkl')

best_arg = grid_search_dict['val']['rmse']['mean'].argmin()
keys, values = zip(*param_grid.items())
all_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
best_params = all_params[best_arg]

df_Xy = df_observation.set_index('Datetime').join([df_context.set_index('Datetime'), df_exogenous.set_index('Datetime')])
df_Xy_train = df_Xy[start_datetime:end_datetime]
Xtrain, ytrain_list, Xnames, days = create_xy_dataset(df_Xy_train, time_series, features_exogenous, features_context)

rf_list = []
for ytrain in tqdm(ytrain_list):
    rf = RandomForestRegressor(**best_params, verbose=0)
    rf.fit(Xtrain,ytrain)
    rf_list.append(rf)
    
# Save models
#save_pickle('/home/toque/data2/forecast/model/rf_uni_inverted/optimize/'+model_name+'/list_rf_uni_inverted.pkl', rf_list)   


# # Predict


start_datetime, end_datetime = '2015-01-01 00:00:00', '2017-12-31 23:45:00'
df_Xy_test = df_Xy[start_datetime:end_datetime]
Xtest, ytest_list, Xnames, days_test = create_xy_dataset(df_Xy_test, time_series, features_exogenous, features_context)


path_directory_to_save = '/home/toque/data2/forecast/model/rf_uni_inverted/prediction/'+model_name+'/'
pred_list = []
for rf in tqdm(rf_list):
    pred_list.append(rf.predict(Xtest))
pred_list = np.array(pred_list)


df_res = pred_list_to_dataframe(pred_list, time_series, days_test)

if not os.path.exists(path_directory_to_save):
    os.makedirs(path_directory_to_save)


#Detrend data to delete
df_res = (df_res.set_index('Datetime') + df_res[['Datetime']].set_index('Datetime').join(pred_lt_rf_uni_inverted.set_index('Datetime'))).reset_index()


df_res.to_csv(path_directory_to_save + start_datetime[:10] + "_" + end_datetime[:10] + '.csv', index=False)


