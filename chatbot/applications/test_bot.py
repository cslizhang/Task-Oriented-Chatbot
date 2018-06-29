# coding:utf8
# @Time    : 18-6-25 下午2:41
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import sys
sys.path.append("/home/zhouzr/project/Task-Oriented-Chatbot")
from chatbot.utils.log import get_logger
logger = get_logger("test")
from chatbot.intent.models.fast_text import FastText
from chatbot.intent.rules.rule_v1 import IntentRuleV1
from chatbot.ner.rules.rule_ner import NerRuleV1
from chatbot.cparse.label import IntentLabel
from chatbot.cparse.vocabulary import Vocabulary
from chatbot.skills.simple import SayHi, Thanks, Praise, Criticize, GoodBye
from chatbot.skills.botQA import BotQA
from chatbot.skills.wanyi import CompanyInfo, CompanyServe
from chatbot.skills.data_query import DataQuery
from chatbot.skills.help import Help
from chatbot.skills.safe import SafeResponse
from chatbot.skills.tuling import Tuling
from chatbot.skills.leave_message import LeaveMessage
from chatbot.skills.file_retrieval import FileRetrieval
from chatbot.bot import ChatBot
from chatbot.utils.path import MODEL_PATH
from wxpy import Bot



intent_model = FastText.load(str(MODEL_PATH/"v0.2"/"intent_model.FastText"))
intent_rule = IntentRuleV1()
ner = NerRuleV1()
label = IntentLabel.load(str(MODEL_PATH/"v0.2"/"label"))
vocab = Vocabulary.load(str(MODEL_PATH/"v0.2"/"vocab"))
file_retrieval = FileRetrieval(str(MODEL_PATH/"v0.2"/"file_retrieval"/"tfidf"),
                          str(MODEL_PATH/"v0.2"/"file_retrieval"/"cluster_index"),
                          str(MODEL_PATH/"v0.2"/"file_retrieval"/"policy_file.utf8.csv"))

cbot = ChatBot(
    intent_model=intent_model,
    intent_rule=intent_rule,
    vocab=vocab,
    label=label,
    ner=ner,
    intent2skills={
        "未知": SafeResponse(),
        "留言": LeaveMessage(),
        "帮助": Help(),
        "闲聊": Tuling(),
        "文件检索": file_retrieval,
        "数据查询": DataQuery(),
        "公司咨询": CompanyInfo(),
        "业务咨询": CompanyServe(),
        "chatbot": BotQA(),
        "表扬": Praise(),
        "批评": Criticize(),
        "打招呼": SayHi(),
        "再见": GoodBye(),
        "感谢": Thanks(),
    }
)
#
# # 初始化机器人，扫码登陆
# bot = Bot()
# my_friend = bot.friends()
# my_groups = bot.groups()
# print(my_groups)
# print(my_friend)
#
# @bot.register(my_groups)
# def wxchat(msg):
#     query={
#         "text": msg.text,
#         "user": msg.member.name,
#         "right": [],
#         "app": "wechat",
#     }
#     return cbot.chat(query)
#
# bot.join()



queries = [
    {
        "user": "zhouzr",
        "text": "昨天用电",
        "right": ["XM001"],
        "app": "test"
    },
    {
        "user": "zhouzr",
        "text": "呵呵，今天",
        "right": ["XM001"],
        "app": "test"
    },
    {
        "user": "zhouzr",
        "text": "前天",
        "right": ["XM001"],
        "app": "test"
    },
    {
        "user": "zhouzr",
        "text": "你好",
        "right": ["XM001"],
        "app": "test"
    },
{
        "user": "zhouzr",
        "text": "四川政策文件",
        "right": ["XM001"],
        "app": "test"
    }
]
for i in queries:
    print(cbot.chat(i))
