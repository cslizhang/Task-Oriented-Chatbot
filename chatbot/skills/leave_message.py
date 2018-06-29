# coding:utf8
# @Time    : 18-6-25 下午2:35
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.skill import BaseSkill


class LeaveMessage(BaseSkill):
    def __call__(self, context):
        return "小益已经帮您记下啦"

    def contain_slots(self, entities):
        return False

    def init_slots(self):
        return {}
