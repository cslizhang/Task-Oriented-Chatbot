# coding:utf8
# @Time    : 18-6-11 上午11:31
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import datetime as dt


class Entity(object):

    @classmethod
    def name(cls):
        return cls.__name__


class TimeInterval(dict, Entity):

    def __init__(self):
        now = dt.datetime.now().date()
        super().__init__(
            start=str(dt.datetime(year=now.year, month=1, day=1).date()),
            end=str(now)
        )


class Location(dict, Entity):

    def __init__(self):
        super().__init__(
            province=None,
            city=None
        )


if __name__ == "__main__":
    t = TimeInterval()
    l = Location()
    print(l)
    print(t)
