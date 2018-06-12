# coding:utf8
# @Time    : 18-6-11 下午4:30
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.bot import Bot
from chatbot.intent.models.fast_text import FastText
from chatbot.intent.rules.rule_v1 import IntentRuleV1
from chatbot.ner.rules.rule_ner import NerRuleV1
from chatbot.skills.tuling import Tuling
from chatbot.skills.file_retrieval import FileRetrieval


intent_model = FastText.load("/home/zhouzr/project/Task-Oriented-Chatbot/chatbot/results/intent/test.FastText")
intent_rule = IntentRuleV1()
ner = NerRuleV1()
tuling = Tuling()