# coding:utf8
# @Time    : 18-6-7 上午10:59
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com


def intent_rule_v1(context):
    query = context["query"]
    if query.startswith("留言"):
        return "留言"
    if query.startswith("意图"):
        return "意图"
    if query.startswith("闲聊"):
        return "闲聊"
