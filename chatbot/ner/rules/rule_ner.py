# coding:utf8
# @Time    : 18-6-7 上午10:51
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.estimator import Estimator
from chatbot.core.entity import TimeInterval


class NerRuleV1:
    def __init__(self):
        super().__init__()

    def extract(self, context):
        """

        :param context: context["query"]
        :return: <dict of list>
        {"TimeInterval": ["", ""]}

        """
        rst = {}
        ext_time = self._extract_time(context)
        if ext_time is not None:
            rst["TimeInterval"] = ext_time

        return rst

    def _extract_time(self, context):
        """

        :param context:
        :return: 假如没有实体，返回None，
        """
        pass

    def transform(self):
        pass

    def _infer_time_entity(self, context):
        """

        :param context:
        :return: <list of time entity>
        """
        # TODO: linjing
        pass

    def _infer_location_entity(self, context):
        pass
