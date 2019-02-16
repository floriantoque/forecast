# Python notes:
# string to datetime format : dateutil.parser.parse(my_string)
# what day is it?: calendar.weekday(libdt.year, libdt.month, libdt.day) (0=monday, 1=tuesday)

import datetime as libdt
import pytz
import pandas as pd

def build_timestamp_list(start, end, time_step_second=15*60):
    """
            Build a list of date between start and end with interval of time_step_second
            e.g., start = 2015-01-01 00:00:00, end = 2015-01-01 01:00:00, time_step_second = 30*60
            return : [2015-01-01 00:00:00, 2015-01-01 00:30:00, 2015-01-01 01:00:00]

            :param start: starting date with format "%Y-%m-%d %H:%M:%S"
            :param end: ending date with format "%Y-%m-%d %H:%M:%S"
            :param time_step_second:  second interval
            :return: list of date with format "%Y-%m-%d %H:%M:%S"

    """
    timestamp_list = [start]

    start = libdt.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end = libdt.datetime.strptime(end, "%Y-%m-%d %H:%M:%S") #+ libdt.timedelta(minutes=time_step_second)

    while start < end:
        start = start + libdt.timedelta(0, time_step_second)
        timestamp_list.append(libdt.datetime.strftime(start, "%Y-%m-%d %H:%M:%S"))
    return timestamp_list


def next_timestamp(start, time_step_second=15*60):
    """
    :param start: timestamp start with string format "%Y-%m-%d %H:%M:%S"
    :param time_step_second: interval between returned value and timestamp start
    :return: next timestamp with interval time_step_second with string format "%Y-%m-%d %H:%M:%S"
    """
    new_dt = libdt.datetime.strptime(start, "%Y-%m-%d %H:%M:%S") + libdt.timedelta(0, time_step_second)
    return libdt.datetime.strftime(new_dt, "%Y-%m-%d %H:%M:%S")

def string_to_datetime(date, hms=True):
    """
    :param date: String date with format "%Y-%m-%d %H:%M:%S" or "%Y-%m-%d" that depends on hms choice
    :param hms: Boolean True mean date with format "%Y-%m-%d %H:%M:%S" else date with format "%Y-%m-%d"
    :return: return date with format datetime
    """
    if hms:
        return libdt.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    else:
        return libdt.datetime.strptime(date, "%Y-%m-%d")


def datetime_to_string(dt, hms=True):
    """
    :param dt: Day with format datetime
    :param hms: Boolean, if True "%Y-%m-%d %H:%M:%S" is returned
    :return: String of the date with format: "%Y-%m-%d", if hms True string of the date with format: "%Y-%m-%d %H:%M:%S"
    """
    if hms:
        return str(dt.year) + "-" + str(dt.month) + "-" +str(dt.day) + " " + str(dt.hour) + ":" + str(dt.minute) + ":" + str(dt.second)
    else:
        return str(dt.year) + "-" + str(dt.month) + "-" +str(dt.day)


def ts_to_id_15min(ts):
    return ts[:10] + "-" + ts[11:13] + "-" + str(int(ts[14:16])/15)


def ts_to_id_30min(ts):
    return ts[:10] + "-" + ts[11:13] + "-" + str(int(ts[14:16])/30)


def ts_to_id_60min(ts):
    return ts[:10] + "-" + ts[11:13]


def bool_h(ts, i):
    if int(ts[11:13]) == i:
        return 1
    else:
        return 0


def bool_qh(ts, i):
    if int(ts[14:16])/15 == i:
        return 1
    else:
        return 0


def bool_dh(ts, i):
    if int(ts[14:16])/30 == i:
        return 1
    else:
        return 0


def get_utc_dt(date_dt, timezone="Europe/Paris"):
    return pytz.timezone(timezone).localize(date_dt, is_dst=None).astimezone(pytz.utc).replace(tzinfo=None)


def unix_time(date_dt):
    epoch = libdt.datetime.utcfromtimestamp(0)
    delta = date_dt - epoch
    return delta.total_seconds()


def unix_time_millis(date_dt):
    return int(unix_time(date_dt) * 1000)


def get_day_string(day_id, fr=True):
    dict_fr={0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}
    dict_en={0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    if fr:
        return dict_fr[day_id]
    return dict_en[day_id]

def get_list_common_date(start, end, obs, list_df_pred):
    df = pd.concat([obs[['Datetime']].set_index('Datetime')[start:end]]+
                   [df_[['Datetime']].set_index('Datetime') for df_ in list_df_pred], axis=1, join='inner')
    return df.index.values.astype(str)

