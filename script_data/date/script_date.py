# example execution:
# python script_date.py --path_save='../../data/date_file_2013_2018_included_test.csv' --start_date="2017-01-01 00:00:00"
# --end_date="2017-07-01 00:00:00" --time_step_second=3600

"""
    Return csv file with following header and values:
    'Datetime': String with format "%Y-%m-%d %H:%M:%S"
    'Day_id': int {0-6}
    'Day_fr': String {Lundi, Mardi, ...}
    'Day_en': String {Monday, Tuesday, ...}
    'Friday': Boolean
    'Monday': Boolean
    'Saturday': Boolean
    'Sunday': Boolean
    'Thursday': Boolean
    'Tuesday': Boolean
    'Wednesday': Boolean
    'School_holidays_france_zoneC': Boolean
    'Extra_day_off_france': Boolean
    'Holidays_france': Boolean
    'Month_id': {1-12}
    'Month_en': String {January, February, ...}
    'April': Boolean
    'February': Boolean
    'January': Boolean
    'July': Boolean
    'June': Boolean
    'March': Boolean
    'May': Boolean
    'hour_minute_second': String with format  "%H:%M:%S"
    'hour_minute_second_numerical': Int {0- (24h/numberoftimestep)-1} if timestep=1h => {0-23}
    '0 - (24h/numberoftimestep)-1': Boolean
    'Category2':String {DIJFP_2, JMHV_2, JOVS_2, LHV_2, MHV_2, SAHV_2, SAVS_2, VHV_2}
    'DIJFP_2': Boolean
    'JMHV_2': Boolean
    'JOVS_2': Boolean
    'LHV_2': Boolean
    'MHV_2': Boolean
    'SAHV_2': Boolean
    'SAVS_2': Boolean
    'VHV_2': Boolean
    'Day_category2_id': int {0,7}
    'Day_category2_en':
    'Category1': DIJFP, JOHV, JOVS, SAHV, SAVS
    'DIJFP': Boolean
    'JOHV': Boolean
    'JOVS': Boolean
    'SAHV': Boolean
    'SAVS': Boolean
    'Day_category1_id': int {0-5}
    'Day_category1_en': String {}
    '24_december': Boolean
    '31_december': Boolean
"""

import numpy as np
import pandas as pd
import datetime as libdt
from tqdm import tqdm
import argparse
import sys
sys.path.insert(0, '../../utils/')
from utils_calendar import *
sys.path.insert(0, '../../utils/')
from utils_date import *




parser = argparse.ArgumentParser(description='Parameters of script date')
parser.add_argument('--path_save', type=str, help='full path used to save csv file of date',
                    default='../../data/date_file_2013_2018_included_test.csv')
parser.add_argument('--start_date', type=str,
                    help="string: start date included with format '%Y-%m-%d %H:%M:%S' example 2017-01-01 00:00:00",
                    default="2017-07-01 00:00:00")
parser.add_argument('--end_date', type=str,
                    help="string: end date included with format '%Y-%m-%d %H:%M:%S' example 2017-07-01 00:00:00",
                    default="2017-07-01 00:00:00")
parser.add_argument('--time_step_second', type=int,
                    help="string: time_step in second",
                    default=3600)
args = parser.parse_args()
path_save = args.path_save
start_date = args.start_date
end_date = args.end_date
time_step_second = args.time_step_second



file_school_holidays = open("../../data/school_holidays_france_zoneC_2007_2018_included.csv", "r")
school_holidays_dt = [(string_to_datetime(i.split(";")[0], hms=False),
                       string_to_datetime(i.split(";")[1], hms=False)) for i in file_school_holidays.read().splitlines()]
file_holidays = open("../../data/holidays_france_2008_2019_included.csv", "r")
holidays_dt = [string_to_datetime(i, hms=False) for i in file_holidays.read().splitlines()]
extra_day_off_dt = get_extra_day_off(holidays_dt, school_holidays_dt)


print("Step 1/2 : Create date features approx 20sec per year")


timestamp_list = build_timestamp_list(start_date, end_date, time_step_second=time_step_second)
df_date = pd.DataFrame(data=timestamp_list, columns=["Datetime"])


df_date["Day_id"] = [string_to_datetime(i).weekday() for i in df_date["Datetime"].values]
df_date["Day_fr"] = [get_day_string(i) for i in df_date["Day_id"].tolist()]
df_date["Day_en"] = [get_day_string(i, fr=False) for i in df_date["Day_id"].tolist()]
df_date = pd.concat([df_date, pd.get_dummies(df_date['Day_en'])], axis=1)


df_date["School_holidays_france_zoneC"] = \
    [int(is_school_holidays(school_holidays_dt, string_to_datetime(i))) for i in tqdm(df_date["Datetime"].values)]
df_date["Extra_day_off_france"] = \
    [int(is_extra_day_off(extra_day_off_dt, string_to_datetime(i))) for i in tqdm(df_date["Datetime"].values)]
df_date["Holidays_france"] = \
    [int(is_holidays(holidays_dt, string_to_datetime(i))) for i in tqdm(df_date["Datetime"].values)]

df_date["Month_id"] = [int(i[5:7]) for i in df_date["Datetime"].values]
df_date["Month_en"] = [string_to_datetime(i).strftime("%B") for i in tqdm(df_date["Datetime"].values)]
df_date = pd.concat([df_date, pd.get_dummies(df_date['Month_en'])], axis=1)


df_date["hour_minute_second"] = [i[11:] for i in df_date["Datetime"].values]
len_hms = len(df_date["hour_minute_second"].drop_duplicates().tolist())
dict_hms_num = dict(zip(df_date["hour_minute_second"].drop_duplicates().tolist(), np.arange(0, len_hms, 1)))
df_date["hour_minute_second_numerical"] = [dict_hms_num[i] for i in df_date["hour_minute_second"].values]
df_date = pd.concat([df_date, pd.get_dummies(df_date["hour_minute_second_numerical"])], axis=1)


df_date["Category2"] = df_date["Datetime"].map(lambda ts: day_to_category2(school_holidays_dt,
                                                          holidays_dt, extra_day_off_dt, string_to_datetime(ts)))
df_date = pd.concat([df_date, pd.get_dummies(df_date['Category2'])], axis=1)
dict_category2 = {"DIJFP_2": 8, "JOVS_2": 5, "SAVS_2": 7, "LHV_2": 1, "JMHV_2": 2, "MHV_2": 4, "VHV_2": 3, "SAHV_2":6}
dict_category2_en = {"DIJFP_2": "SPD_2", "JOVS_2": "W-H_2", "SAVS_2": "S-H_2", "LHV_2": "M-OSH_2", "JMHV_2": "TT-OSH_2", "MHV_2": "W-OSH_2",
                     "VHV_2": "F-OSH_2", "SAHV_2": "S-OSH_2"}
df_date["Day_category2_id"] = [dict_category2[i] for i in df_date["Category2"].values]
df_date["Day_category2_en"] = [dict_category2_en[i] for i in df_date["Category2"].values]


df_date["Category1"] = df_date["Datetime"].map(lambda ts: day_to_category1(school_holidays_dt,
                                                          holidays_dt, extra_day_off_dt, string_to_datetime(ts)))
df_date = pd.concat([df_date, pd.get_dummies(df_date['Category1'])], axis=1)
dict_category1 = {"JOHV": 1, "JOVS": 2, "SAHV": 3, "SAVS": 4, "DIJFP": 5}
dict_category1_en = {"JOHV": "WD-OSH", "JOVS": "WD-H", "SAHV": "S-OSH", "SAVS": "S-H", "DIJFP": "SHEDO"}
df_date["Day_category1_id"] = [dict_category1[i] for i in df_date["Category1"]]
df_date["Day_category1_en"] = [dict_category1_en[i] for i in df_date["Category1"]]

df_date["24_december"] = [int(i[5:10] == "12-24") for i in df_date["Datetime"]]
df_date["31_december"] = [int(i[5:10] == "12-31") for i in df_date["Datetime"]]

print("Step 2/2 : Save file")
df_date.to_csv(path_save, index=False)
