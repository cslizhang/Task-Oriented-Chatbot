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
        labels = [label.rstrip("\n") for label in f.readlines()]
    return labels


intent_labels = get_intent_labels()
logger = get_logger(__name__)


class IntentLabel(Dictionary):
    def __init__(self):
        super().__init__()
        # self.fit(intent_labels)

    def fit(self, x):
        if self.training:
            if isinstance(x, str):
                self._add_one(x)
            elif isinstance(x, list) and isinstance(x[0], str):
                for w in x:
                    self._add_one(w)
            elif isinstance(x, list) and isinstance(x[0], list):
                for s in x:
                    for w in s:
                        self._add_one(w)
            else:
                raise ValueError("input error")
        else:
            logger.info("{} can't training now".format(self.__class__.__name__))


    def transform(self, labels):
        """

        :param labels: <list of string>,每个元素代表一个标签
        :return: <list>
        """
        return [self.transform_one(l) for l in labels]

    def transform_one(self, label):
        return self.word2idx.get(label, UNDEFINE_IDX)

    def reverse(self, labels_ids):
        return [self.idx2word.get(l, UNDEFINE_IDX) for l in labels_ids]

    def reverse_one(self, label_idx):
        return self.idx2word.get(label_idx, UNDEFINE_IDX)


if __name__ == "__main__":
    label = IntentLabel()
    x = ["谢谢", "你好"]
    label.fit(x)
