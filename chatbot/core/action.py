# coding:utf8
# @Time    : 18-6-2 下午1:29
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com


class BaseAction(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return "Action {}".format(self.__class__.__name__)

    def __repr__(self):
        return self.__str__()
