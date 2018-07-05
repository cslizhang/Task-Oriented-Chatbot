# coding:utf8
# @Time    : 18-6-25 下午2:35
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.skill import BaseSkill
from chatbot.utils.path import ROOT_PATH
import pandas as pd
import codecs

class LeaveMessage(BaseSkill):
    def __call__(self, context):
        print(context)
        path = str(ROOT_PATH.parent / "corpus" / "intent" / "skill" / "leavemeasage.txt")
        with codecs.open(path, "a+","utf-8") as f:
            f.write(context['context_id']+'\t'+'\t'+ context['app']+'\t'+'\t'+ context['last_query_time']+'\t'+'\t'+ context['user']+'\t'+'\t'+context['query']+'\n'+'\n')
        f.close()
        return "小益已经帮您记下啦"

    def contain_slots(self, entities):
        return False

    def init_slots(self):
        return {}


if __name__ == "__main__":
    context = {'context_id':'1','app':'益电宝','user':'abc','query':'请你们公司负责市场的同事跟我联系一下，电话1454123','last_query_time':'2018-07-04 15:16'}
    P=LeaveMessage()
    P(context)