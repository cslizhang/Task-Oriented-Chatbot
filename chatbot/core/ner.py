# coding:utf8
# @Time    : 18-6-8 下午3:40
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import abc

from chatbot.core.estimator import Estimator


class BaseNER(Estimator):

    @abc.abstractmethod
    def infer(self, context):
        """提取当前会话的所有实体

        :param context: <class Context>
        :return: <dict of list>, key为实体（槽）名，
        values为槽值的列表，每个槽值为一个dict
        """