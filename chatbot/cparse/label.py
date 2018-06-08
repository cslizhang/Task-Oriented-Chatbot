# coding:utf8
# @Time    : 18-5-21 下午2:10
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.config import UNDEFINE, UNDEFINE_IDX
from chatbot.utils.log import get_logger
from chatbot.cparse.dictionary import Dictionary
logger = get_logger(__name__)


class Label(Dictionary):
    def __init__(self, un_define_idx=0):
        super().__init__()
        self.word2idx = {UNDEFINE: UNDEFINE_IDX}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.idx = len(self.word2idx)

    def transform(self, labels):
        """

        :param labels: <list>
        :return: <list>
        """
        return [self.word2idx.get(l, UNDEFINE_IDX) for l in labels]

    def reverse(self, labels_id):
        return [self.idx2word.get(l, UNDEFINE_IDX) for l in labels_id]