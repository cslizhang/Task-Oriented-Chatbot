# coding:utf8
# @Time    : 18-6-7 上午10:59
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.estimator import Estimator
from chatbot.cparse.label import IntentLabel


class IntentRuleV1(Estimator):
    def __init__(self):
        super().__init__()
        self._intent_label = IntentLabel()

    def infer(self, query):
        """

        :param query: 没有经过转化的原始输入
        :return: <tuple> (class, prob)
        """
        if query.startswith("留言"):
            rst = self._intent_label.transform_one("留言")
            return rst, 1.0
        elif query.startswith("意图"):
            rst = self._intent_label.transform_one("意图语料收集")
            return rst, 1.0
        else:
            return self._intent_label.transform_one("闲聊"), 1.0


if __name__ == "__main__":
    rule = IntentRuleV1()
    print(rule.infer("意图 我"))