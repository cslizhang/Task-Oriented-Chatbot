# coding:utf8
# @Time    : 18-6-4 下午2:58
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com


class BaseSkill(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _check_satisfied(self, *args, **kwargs):
        raise NotImplementedError

    def _act(self, *args, **kwargs):
        raise NotImplementedError
