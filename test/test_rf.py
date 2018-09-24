

import unittest

import sys

sys.path.insert(0, '../')
from model.rf.rf_model import Rf_model
from utils.utils import *

class Rf_test(unittest.TestCase):

    """Test case for test the functions of class 'rf_model'."""

    def test_optimize(self):
        observation_data_path = ['../data/observation_file_2017-01-01_2017-06-30_included_test.csv']
        exogenous_data_path = ['../data/date_file_2013_2018_included_test.csv']
        features_list = [['Day_id', 'Month_id', 'School_holidays_france_zoneC', 'Extra_day_off_france',
                          'Holidays_france', 'hour_minute_second_numerical'], ['hour_minute_second_numerical']]
        time_series = ['71634', '71650', '71442', '71654', '71743', '71328',
                       '71305', '71517', '71284', '415852', '71404', '71298', '73630',
                       '71318', '71348', '71379', '71647', '71663', '71673', '71485',
                       '71222', '71297', '71347', '71100', '71133', '71217', '73696',
                       '73689', '71407', '73616', '70537', '70636', '70375', '71351',
                       '71977', '70452', '72031', '72013', '70645', '71253', '71363',
                       '70596', '72430', '71201', '72460', '70488', '71076', '70604',
                       '73695', '70143', '70248', '71001', '73615']

        model_name = 'rf_model_test'
        start_date = '2017-01-01 00:00:00'
        end_date = '2017-01-05 00:00:00'
        path_to_save = '../data/model/test/rf/'
        param_kfold = {'n_splits': 2, 'shuffle': True, 'random_state': 0}
        param_grid = {'n_estimators': [1], 'max_features': ['auto'], 'max_depth': [None], 'min_samples_split': [5],
                      'min_samples_leaf': [5], 'n_jobs': [6], 'criterion': ['mse']}
        scoring = "neg_mean_squared_error"
        scaler_choice = "standard"


        df_Xy = read_csv_list(observation_data_path).set_index('Datetime').join(
            read_csv_list(exogenous_data_path).set_index('Datetime'))[start_date:end_date]

        X_list = [df_Xy[features].values for features in features_list]
        y = df_Xy[time_series].values

        my_model = Rf_model(model_name, start_date, end_date, features_list, time_series, observation_data_path,
                            exogenous_data_path, scaler_choice)
        grid_search_dict = my_model.optimize(X_list, y, param_grid, param_kfold, scoring)

        best_conf = [(features, grid_search_dict[tuple(features)].best_params_,
                      grid_search_dict[tuple(features)].best_score_) for features in list(grid_search_dict.keys())]
        best_conf.sort(key=lambda x: x[2])
        features = list(best_conf[-1][0])
        best_params = best_conf[-1][1]

        self.assertEqual(best_params['n_estimators'], 1)
        self.assertEqual(best_params['max_features'], 'auto')
        self.assertEqual(best_params['min_samples_leaf'], 5)
        self.assertEqual(best_params['n_jobs'], 6)
        self.assertEqual(best_params['min_samples_split'], 5)
        self.assertEqual(best_params['criterion'], 'mse')
        self.assertEqual(best_params['max_depth'], None)

        return 0


    def test_fit(self):
        observation_data_path = ['../data/observation_file_2017-01-01_2017-06-30_included_test.csv']
        exogenous_data_path = ['../data/date_file_2013_2018_included_test.csv']
        features = ['Day_id', 'Month_id', 'School_holidays_france_zoneC', 'Extra_day_off_france',
                    'Holidays_france', 'hour_minute_second_numerical']
        time_series = ['71634', '71650', '71442', '71654', '71743', '71328',
                       '71305', '71517', '71284', '415852', '71404', '71298', '73630',
                       '71318', '71348', '71379', '71647', '71663', '71673', '71485',
                       '71222', '71297', '71347', '71100', '71133', '71217', '73696',
                       '73689', '71407', '73616', '70537', '70636', '70375', '71351',
                       '71977', '70452', '72031', '72013', '70645', '71253', '71363',
                       '70596', '72430', '71201', '72460', '70488', '71076', '70604',
                       '73695', '70143', '70248', '71001', '73615']

        model_name = 'rf_model_test'
        start_date = '2017-01-01 00:00:00'
        end_date = '2017-01-31 00:00:00'
        path_to_save = '../data/model/test/rf/'
        scaler_choice = 'standard'
        best_params = {'criterion': 'mse', 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 3,
                       'min_samples_split': 5, 'n_estimators': 30, 'n_jobs': 6}


        df_observation = read_csv_list(observation_data_path)
        df_exogenous = read_csv_list(exogenous_data_path)

        df_Xy = df_observation.set_index('Datetime').join(df_exogenous.set_index("Datetime"))[start_date:end_date]
        X = df_Xy[features].values
        y = df_Xy[time_series].values



        my_model = Rf_model(model_name, start_date, end_date, [features], time_series, observation_data_path,
                            exogenous_data_path, scaler_choice)
        my_model.infos['features'] = features
        my_model.infos['best_params'] = best_params

        rf = my_model.fit(X, y, my_model.infos['best_params'])

        feature_importances = dict(zip(features, np.round(rf.feature_importances_ * 100, 2).tolist()))
        my_model.infos["feature_importances"] = feature_importances

        self.assertEqual(my_model.infos['name'], model_name)
        self.assertEqual(my_model.infos['features'], features)
        self.assertEqual(my_model.infos['time_series'], time_series)
        self.assertEqual(my_model.infos['start_date'], start_date)
        self.assertEqual(my_model.infos['end_date'], end_date)
        self.assertEqual(my_model.infos['exogenous_data_path'], exogenous_data_path)
        self.assertEqual(my_model.infos['observation_data_path'], observation_data_path)
        self.assertEqual(my_model.infos['best_params'], best_params)
        self.assertEqual(my_model.infos['feature_importances'], feature_importances)

        return 0

    def test_predict(self):
        observation_data_path = ['../data/observation_file_2017-01-01_2017-06-30_included_test.csv']
        exogenous_data_path = ['../data/date_file_2013_2018_included_test.csv']
        features = ['Day_id', 'Month_id', 'School_holidays_france_zoneC', 'Extra_day_off_france',
                    'Holidays_france', 'hour_minute_second_numerical']
        time_series = ['71634', '71650', '71442', '71654', '71743', '71328',
                       '71305', '71517', '71284', '415852', '71404', '71298', '73630',
                       '71318', '71348', '71379', '71647', '71663', '71673', '71485',
                       '71222', '71297', '71347', '71100', '71133', '71217', '73696',
                       '73689', '71407', '73616', '70537', '70636', '70375', '71351',
                       '71977', '70452', '72031', '72013', '70645', '71253', '71363',
                       '70596', '72430', '71201', '72460', '70488', '71076', '70604',
                       '73695', '70143', '70248', '71001', '73615']

        model_name = 'rf_model_test'
        start_date = '2017-01-01 00:00:00'
        end_date = '2017-12-31 00:00:00'
        path_to_save = '../data/model/test/rf/'
        scaler_choice = 'standard'
        best_params = {'criterion': 'mse', 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 3,
                       'min_samples_split': 5, 'n_estimators': 100, 'n_jobs': 6}

        df_observation = read_csv_list(observation_data_path)
        df_exogenous = read_csv_list(exogenous_data_path)

        df_Xy = df_observation.set_index('Datetime').join(df_exogenous.set_index("Datetime"))[start_date:end_date]
        X = df_Xy[features].values
        y = df_Xy[time_series].values

        my_model = Rf_model(model_name, start_date, end_date, [features], time_series, observation_data_path,
                            exogenous_data_path, scaler_choice)
        my_model.infos['features'] = features
        my_model.infos['best_params'] = best_params

        rf = my_model.fit(X, y, my_model.infos['best_params'])

        feature_importances = dict(zip(features, np.round(rf.feature_importances_ * 100, 2).tolist()))
        my_model.infos["feature_importances"] = feature_importances

        X = df_Xy[my_model.infos['features']].values
        obs = np.around(df_Xy[time_series][start_date:end_date].values, decimals=2)
        pred = my_model.predict(rf, X)
        pred_mean = np.around(df_Xy[time_series][start_date:end_date].values.mean(axis=0), decimals=2)

        self.assertLess(np.abs(pred - obs).mean(), np.abs(pred_mean - obs).mean())

        return 0


unittest.main()
