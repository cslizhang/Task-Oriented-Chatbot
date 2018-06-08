# coding:utf8
# @Time    : 18-6-4 下午2:58
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com


class BaseSkill(object):

    def __call__(self, context):
        """返回文本

        :param context: <class Context>
        :return: response <String>
        """
        raise NotImplementedError

    def _check_satisfied(self, context):
        """检查词槽是否满足

        :param context: <class Context>
        :return: label <bool> 是否满足的标签, question <String> 不满足的回复
        """
        raise NotImplementedError

    def _act(self, context):
        """词槽满足的回复

        :param context:  <class Context>
        :return: response <String>
        """
        raise NotImplementedError

    def check_update_slots(self, entities):
        """检查是否可以更新词槽
        :param entities:  <dict>
        :return: label <bool> 是否可以更新标签
        """
        raise NotImplementedError

    @property
    def name(self):
        """skill名称"""
        raise NotImplementedError

    @property
    def init_slots(self):
        """初始化默认词槽"""
        raise NotImplementedError
