# coding:utf8
# @Time    : 18-6-25 下午2:36
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.skill import BaseSkill


class BotQA(BaseSkill):
    def __call__(self, context):
        return "小益的相关信息回复..."

    def contain_slots(self, entities):
        return False

    def init_slots(self):
        return {}
