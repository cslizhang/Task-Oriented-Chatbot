# coding:utf8
# @Time    : 18-6-7 上午10:51
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com


class RuleNer(object):
    def __init__(self):
        pass

    def predict(self, text):
        r1 = {
            "location": {"city": "乐山", "province": "全国"},
            "time_interval": {"start": "2018-01-02", "end": "2018-05-01"}
        }
        return r1
