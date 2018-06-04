# coding:utf8
# @Time    : 18-6-4 下午4:15
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import datetime as dt


def get_timeiterval_slot():
    return {
        "start_year": None,
        "start_month": None,
        "start_dayofmonth": None,
        "end_year": None,
        "end_month": None,
        "end_dayofmonth": None,
    }


def get_loation_slot():
    return {
        "province": None,
        "city": None,
    }


class Time(object):
    def __init__(self, year=None, month=None, dayofmonth=None):
        self.year = year
        self.month = month
        self.dayofmonth = dayofmonth
        self.fill_by_default()

    def fill_by_default(self):
        now = dt.datetime.now().date()
        self.year = now.year
        self.month = now.month
        self.dayofmonth = now.day

    @property
    def is_ok(self):
        if any([
            self.year is None,
            self.month is None,
            self.dayofmonth is None
        ]):
            return False
        else:
            return True


class TimeInterval(object):
    def __init__(self,):
        now = dt.datetime.now()
        self.start = now
        self.end = now

    def update(self, year=None, month=None, dayofyear=None):
        pass

    def _update_year(self, year):
        if len(year) == 1:
            self.start = self.start
            self.end.year = year
        else:
            pass



