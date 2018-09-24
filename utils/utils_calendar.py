import numpy as np
import calendar
import dateutil.parser
import datetime as libdt




CATEGORY1 = ['JOHV', 'SAHV', 'JOVS', 'SAVS', 'DIJFP']
CATEGORY2 = ['LHV', 'MHV', 'JMHV', 'VHV', 'SAHV', 'JOVS', 'SAVS', 'DIJFP']

def get_extra_day_off(holidays_dt,school_holidays_dt):
    """
        :param holidays_dt: List of holidays with format datetime
        :param school_holidays_dt: List of school holidays with format datetime
        :return: List of extra day off with format datetime
    """
    extra_day_off=[]
    for h in holidays_dt:
        if h.weekday() == 3:
            day_next = h+libdt.timedelta(days=1)
            if not(is_school_holidays(school_holidays_dt, day_next)):
                extra_day_off.append(day_next)
        elif h.weekday() == 1:
            day_prev = h+libdt.timedelta(days=-1)
            if not(is_school_holidays(school_holidays_dt, day_prev)):
                extra_day_off.append(day_prev)
    return extra_day_off


def is_school_holidays(school_holidays_dt, day_dt):
    """
        :param school_holidays_dt: List of tuple(start_date, end_date) which represents the school holidays period,
               format of the dates is datetime
        :param day_dt: Day with format datetime to analyze
        :return: Boolean, True if day is a school holidays False if not
    """

    for sh in school_holidays_dt:
        start = sh[0].date()
        end = sh[1].date()
        if (day_dt.date() >= start) & (day_dt.date() <= end):
            return True
    return False


def is_holidays(holidays_dt, day_dt):
    """
        :param holidays_dt: List of holidays with format datetime
        :param day_dt: Day with format datetime to analyze
        :return: True if day is a holiday, False if not
    """
    holidays_dt = [i.date() for i in holidays_dt]
    return day_dt.date() in holidays_dt


def is_extra_day_off(extra_day_off_dt, day_dt):

    """
        Extra day off is a day off between a holiday and the weekend
        :param extra_day_off_dt: List of extra day off with format datetime
        :param day_dt: Day with format datetime to analyze
        :return: True if day is an extra day off, False if not
    """
    extra_day_off_dt = [i.date() for i in extra_day_off_dt]
    return day_dt.date() in extra_day_off_dt


def day_to_category1(school_holidays_dt, holidays_dt, extra_day_off_dt, day_dt):
    """
        # CATEGORY 1:
        # JOHV: jour ouvre, hors vacances (eng: Working days, not in school holidays)
        # SAHV: samedi, hors vacances (eng: Saturday, not in school holidays)
        # JOVS: jour ouvre, vacances (eng: Working days, during school holidays)
        # SAVS: samedi, vacances (eng: Saturday, during school holidays)
        # DIJFP: dimanche, jours feries, ponts (eng: Sunday, holidays and extra day off)

        :param school_holidays_dt: List of tuple(start_date, end_date) which represents the school holidays period,
               format of the dates is datetime
        :param holidays_dt: List of holidays with format datetime
        :param extra_day_off_dt: List of extra day off with format datetime
        :param day_dt: Day with format datetime to analyze
        :return: The category of the day 'day_dt'
    """

    if (day_dt.weekday() == 6) | (is_holidays(holidays_dt, day_dt)) | (is_extra_day_off(extra_day_off_dt, day_dt)):
        return "DIJFP"
    else:
        if is_school_holidays(school_holidays_dt, day_dt):
            if day_dt.weekday()==5:
                return "SAVS"
            else:
                return "JOVS"
        else:
            if day_dt.weekday()==5:
                return "SAHV"
            else:
                return "JOHV"


def day_to_category2(school_holidays_dt, holidays_dt, extra_day_off_dt, day_dt):
    """
        # CATEGORY 2:
        # LHV: lundi hors vacances (eng: Monday, not in school holidays)
        # MHV: mercredi hors vacances (eng: Wednesday, not in school holidays)
        # JMHV: jeudi et mardi hors vacances (eng: Thursday or Tuesday, not in school holidays)
        # VHV: vendredi hors vacances (eng: Friday, not in school holidays)
        # SAHV: samedi, hors vacances (eng: Saturday, not in school holidays)
        # JOVS: days ouvre, vacances (eng: Working days, during school holidays)
        # SAVS: samedi, vacances (eng: week end, in school holidays)
        # DIJFP: dimanche, jour ferie, pont (eng: Sunday, holidays and extra day off)

        :param school_holidays_dt: List of tuple(start_date, end_date) which represents the school holidays period,
               format of the dates is datetime
        :param holidays_dt: List of holidays with format datetime
        :param extra_day_off_dt: List of extra day off with format datetime
        :param day_dt: Day with format datetime to analyze
        :return: The category of the day 'day_dt'
    """

    if (day_dt.weekday() == 6) | (is_holidays(holidays_dt, day_dt)) | (is_extra_day_off(extra_day_off_dt, day_dt)):
        return "DIJFP_2"
    else:
        if is_school_holidays(school_holidays_dt,day_dt):
            if day_dt.weekday() == 5:
                return "SAVS_2"
            else:
                return "JOVS_2"
        else:
            if day_dt.weekday() == 5:
                return "SAHV_2"
            else:
                if day_dt.weekday() == 0:
                    return "LHV_2"
                else:
                    if (day_dt.weekday() == 1) | (day_dt.weekday() == 3):
                        return "JMHV_2"
                    else:
                        if day_dt.weekday() == 4:
                            return "VHV_2"
                        else:
                            return "MHV_2"


def day_to_string_en(day_dt):
    """
        :param day_dt: Day with format datetime
        :return: English word of the day (e.g., Monday, Tuesday, ...)
    """
    dict_res = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    return dict_res.get(day_dt.weekday(), "Invalid index of day")








