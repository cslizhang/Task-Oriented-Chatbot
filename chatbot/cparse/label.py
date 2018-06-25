# coding:utf8
# @Time    : 18-5-21 下午2:10
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.config.constant import UNDEFINE, UNDEFINE_IDX
from chatbot.utils.log import get_logger
from chatbot.cparse.dictionary import Dictionary
from chatbot.utils.path import ROOT_PATH


def get_intent_labels():
    with open(str(ROOT_PATH / "config" / "intent.cfg"), "r") as f:
        labels = [l.rstrip("\n") for l in f.readlines()]
    return labels


intent_labels = get_intent_labels()
logger = get_logger(__name__)


class IntentLabel(Dictionary):
    def __init__(self):
        # TODO: init class for config intent
        super().__init__()

    def fit(self, x):
        """

        :param x: string or list of string, string means a label
        :return:
        """
        if self.training:
            if isinstance(x, str):
                self._add_one(x)
            elif isinstance(x, list) and isinstance(x[0], str):
                for w in x:
                    self._add_one(w)
            else:
                raise ValueError("input error")
        else:
            logger.info("{} can't training now".format(self.__class__.__name__))

    def transform(self, x):
        """

        :param x: <list of string>,每个元素代表一个标签
        :return: <list>
        """
        assert isinstance(x, list), ValueError
        return [self.transform_one(i) for i in x]

    def transform_one(self, x):
        """

        :param x: string
        :return:
        """
        return self.word2idx.get(x, UNDEFINE_IDX)

    def reverse(self, x):
        """

        :param x: list of int
        :return:
        """
        assert isinstance(x, list) and isinstance(x[0], int)
        return [self.reverse_one(l) for l in x]

    def reverse_one(self, x):
        """

        :param x:int
        :return:
        """
        return self.idx2word.get(x, UNDEFINE)


if __name__ == "__main__":
    label = IntentLabel()
    s = ["谢谢", "你好"]
    label.fit(s)
    print(label.transform(s))
    print(label.transform_one(s[0]))
