# -*- coding: utf-8 -*-
# @Time    : 2018/7/4 15:05
# @Author  : Lin
# @File    : chatRecord.py

from chatbot.core.skill import BaseSkill
from chatbot.utils.path import ROOT_PATH
import pandas as pd
import codecs
from chatbot.bot import ChatBot
from chatbot.ner.rules.rule_ner import NerRuleV1


def chat_record(context):
    """
    聊天记录存储到txt
    :param context: context
    :return: None
    """
    path = str(ROOT_PATH.parent / "corpus" / "intent" / "skill" / "Chart_record.txt")
    with codecs.open(path, "a+", "utf-8") as f:
        f.write(context['context_id'] + '\t' + '\t' + context['app'] + '\t' + '\t' + context['last_query_time'] + '\t'
                + '\t' + context['intent'] + '\t' + '\t' + context['slots'] + '\t' + '\t' + context['user'] + '\t'
                + '\t' + context['query'] + '\t' + '\t' + context['history_resp'][-1] + '\n' + '\n')
        f.close()


if __name__ == "__main__":
    test = {'context_id': '1', 'app': '益电宝', 'user': 'abc', 'query': '请你们公司负责市场的同事跟我联系一下，电话14541234654',
            'last_query_time': '2018-07-04 15:16', 'intent': '留言',
            'slots': "{'TimeInterval': {'start': '1454-12-34', 'end': '1454-12-34'}}",
            'history_resp': ['你好', '小益已经帮你记下来']}
    chat_record(test)
