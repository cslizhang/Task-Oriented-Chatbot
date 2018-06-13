# coding:utf8
# @Time    : 18-6-7 上午10:51
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.estimator import Estimator


class NerRuleV1(Estimator):
    def __init__(self):
        super().__init__()

    def infer(self, context):
        """

        :param context:
        :return: <dict of list> {"TimeInterval": [t1, t2, ...], "Location": []}
        """
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




