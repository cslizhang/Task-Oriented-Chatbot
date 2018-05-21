# coding:utf8
# @Time    : 18-5-21 下午2:10
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.cparse.constant import *
from chatbot.utils.log import get_logger
from chatbot.cparse.base import Dictionary


logger = get_logger(__name__)


class Label(Dictionary):
    def __init__(self):
        super().__init__()

    def transform(self, labels):
        return [self.word2idx.get(l, 0) for l in labels]

    def reverse(self, labels_id):
        return [self.idx2word.get(l, 0) for l in labels_id]