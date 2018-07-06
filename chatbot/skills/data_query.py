# coding:utf8
# @Time    : 18-6-25 下午2:36
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.skill import BaseSkill
from chatbot.core.entity import TimeInterval, Company


class DataQuery(BaseSkill):
    def __call__(self, context):
        t = context["slots"][context["intent"]][TimeInterval.name()]
        return "您在查询{}至{}的用电数据".format(
            t["start"],
            t["end"]
        )

    def contain_slots(self, entities):
        """

        :param entities: <dict of list>, key: entity name, values: list of entity
        :return:
        """
        for k, v in entities.items():
            if k in self.init_slots().keys():
                return True
        return False

    def init_slots(self):
        return {TimeInterval.name(): TimeInterval(), Company.name(): Company()}
