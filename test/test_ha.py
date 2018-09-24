

import unittest

import sys

sys.path.insert(0, '../')
from model.ha.ha_model import Ha_model
from utils.utils import *

class Ha_test(unittest.TestCase):

    """Test case for test the functions of class 'ha_model'."""

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

        model_name = 'ha_model_test'
        start_date = '2017-01-01 00:00:00'
        end_date = '2017-01-01 01:00:00'
        path_to_save = '../data/model/test/ha/'

        my_model = Ha_model(model_name, start_date, end_date, features, time_series, observation_data_path,
                            exogenous_data_path)

        df_observation = read_csv_list(observation_data_path)
        df_exogenous = read_csv_list(exogenous_data_path)

        df_Xy = df_observation.set_index('Datetime').join(df_exogenous.set_index("Datetime"))[start_date:end_date]

        X = df_Xy[features].values
        y = df_Xy[time_series].values

        my_model.fit(X, y)

        obs1 = np.around(df_observation.set_index('Datetime').loc['2017-01-01 00:00:00'].values, decimals=2)
        res1 = np.around(my_model.infos['dict_pred_mean'][(6, 1, 1, 0, 1, 0)].astype(float), decimals=2)
        obs2 = np.around(df_observation.set_index('Datetime').loc['2017-01-01 01:00:00'].values, decimals=2)
        res2 = np.around(my_model.infos['dict_pred_mean'][(6, 1, 1, 0, 1, 1)].astype(float), decimals=2)

        self.assertEqual((obs1 != res1).sum(), 0)
        self.assertEqual((obs2 != res2).sum(), 0)
        self.assertEqual(my_model.infos['name'], 'ha_model_test')
        self.assertEqual(my_model.infos['features'], features)
        self.assertEqual(my_model.infos['time_series'], time_series)
        self.assertEqual(my_model.infos['start_date'], start_date)
        self.assertEqual(my_model.infos['end_date'], end_date)
        self.assertEqual(my_model.infos['exogenous_data_path'], exogenous_data_path)
        self.assertEqual(my_model.infos['observation_data_path'], observation_data_path)

        return 0

    def test_save(self):

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

        model_name = 'ha_model_test'
        start_date = '2017-01-01 00:00:00'
        end_date = '2017-01-01 01:00:00'
        path_to_save = '../data/model/test/ha/'

        my_model = Ha_model(model_name, start_date, end_date, features, time_series, observation_data_path,
                            exogenous_data_path)

        df_observation = read_csv_list(observation_data_path)
        df_exogenous = read_csv_list(exogenous_data_path)

        df_Xy = df_observation.set_index('Datetime').join(df_exogenous.set_index("Datetime"))[start_date:end_date]

        X = df_Xy[features].values
        y = df_Xy[time_series].values

        my_model.fit(X, y)

        my_model.save(path_to_save, None)

        load_model = load_pickle(path_to_save + model_name + '.pkl')

        obs1 = np.around(df_observation.set_index('Datetime').loc['2017-01-01 00:00:00'].values, decimals=2)
        res1 = np.around(load_model.infos['dict_pred_mean'][(6, 1, 1, 0, 1, 0)].astype(float), decimals=2)
        obs2 = np.around(df_observation.set_index('Datetime').loc['2017-01-01 01:00:00'].values, decimals=2)
        res2 = np.around(load_model.infos['dict_pred_mean'][(6, 1, 1, 0, 1, 1)].astype(float), decimals=2)

        self.assertEqual((obs1 != res1).sum(), 0)
        self.assertEqual((obs2 != res2).sum(), 0)
        self.assertEqual(load_model.infos['name'], 'ha_model_test')
        self.assertEqual(load_model.infos['features'], features)
        self.assertEqual(load_model.infos['time_series'], time_series)
        self.assertEqual(load_model.infos['start_date'], start_date)
        self.assertEqual(load_model.infos['end_date'], end_date)
        self.assertEqual(load_model.infos['exogenous_data_path'], exogenous_data_path)
        self.assertEqual(load_model.infos['observation_data_path'], observation_data_path)

        return 0

    def test_predict(self):

        model_name = 'ha_model_test'
        path_to_save = '../data/model/test/ha/'
        load_model = load_pickle(path_to_save + model_name + '.pkl')
        exogenous_data_path = load_model.infos['exogenous_data_path']
        list_date = ['2017-01-01 00:00:00', '2017-01-01 01:00:00']

        observation_data_path = ['../data/observation_file_2017-01-01_2017-06-30_included_test.csv']
        df_observation = read_csv_list(observation_data_path)
        X = read_csv_list(exogenous_data_path).set_index('Datetime').ix[list_date][load_model.infos['features']].values
        pred_mean = load_model.predict(X, choice='mean')
        pred_median = load_model.predict(X, choice='median')

        res1 = np.around(pred_mean[0], decimals=2)
        res2 = np.around(pred_mean[1], decimals=2)
        res1_ = np.around(pred_median[0], decimals=2)
        res2_ = np.around(pred_median[1], decimals=2)
        obs1 = np.around(df_observation.set_index('Datetime').loc['2017-01-01 00:00:00'].values, decimals=2)
        obs2 = np.around(df_observation.set_index('Datetime').loc['2017-01-01 01:00:00'].values, decimals=2)

        self.assertEqual((obs1 != res1).sum(), 0)
        self.assertEqual((obs2 != res2).sum(), 0)
        self.assertEqual((obs1 != res1_).sum(), 0)
        self.assertEqual((obs2 != res2_).sum(), 0)

        return 0


unittest.main()
