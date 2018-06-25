# coding:utf8
# @Time    : 18-5-21 上午11:46
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import numpy as np

from chatbot.config.constant import *
from chatbot.utils.log import get_logger
from chatbot.cparse.dictionary import Dictionary
logger = get_logger(__name__)


class Vocabulary(Dictionary):
    def __init__(self):
        super(Vocabulary, self).__init__()
        self.word2idx = {PAD: PAD_IDX, UNK: UNK_IDX}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.idx = len(self.word2idx)

    def fit(self, x):
        """

        :param x: string, list of string, list of list of string
        :return: id or list of id, or list of list of id
        """
        if self.training:
            if isinstance(x, str):
                self._add_one(x)
            elif isinstance(x, list):
                if isinstance(x[0], str):
                    for w in x:
                        self._add_one(w)
                elif isinstance(x[0], list):
                    for s in x:
                        for w in s:
                            self._add_one(w)
                else:
                    raise ValueError
            else:
                raise ValueError("input error")
        else:
            logger.info("{} can't training now".format(self.__class__.__name__))

    def _transform_input_error(self):
        raise ValueError("required list of string or list of list of string")

    def transform(self, x, max_length=None):
        """文本转换成id

        :param x: string of int or <list of list of string> 最小元素为string,代表一个单词
        :param max_length: 固定返回的每个句子固定长度
        :return: <list of list> 最小元素为单词的idx
        """
        if isinstance(x, list):
            if isinstance(x[0], str):
                return self.transform_one(x)
            elif isinstance(x[0], list):
                if isinstance(x[0][0], str):
                    rst = []
                    for sentence in x:
                        rst_s = self.transform_one(sentence, max_length=max_length)
                        rst.append(rst_s)
                    return rst
                else:
                    self._transform_input_error()
            else:
                self._transform_input_error()
        else:
            raise ValueError("required list of string or list of list of string")

    def transform_one(self, x, max_length=None):
        rst_s = [self.word2idx.get(word, UNK_IDX) for word in x]
        if max_length is not None:
            rst_s = rst_s[:max_length]
            rst_s = [PAD_IDX] * (max_length - len(rst_s)) + rst_s
        return rst_s

    def reverse_one(self, x):
        """

        :param x:
        :return:
        """
        rst = []
        for idx in x:
            rst.append(self.idx2word.get(idx, UNK_IDX))
        rst = " ".join(rst)
        return rst

    def reverse(self, x):
        """

        :param x: <list of list>
        :return:
        """
        rst = []
        for sentence in x:
            rst_i = []
            for idx in sentence:
                rst_i.append(self.idx2word.get(idx, UNK_IDX))
            rst_i = " ".join(rst_i)
            rst.append(rst_i)
        return rst


def get_embedding_matrix(model, dictionary):
    matrix = []
    miss_match = 0
    idx2word_sorted = sorted(dictionary.word2idx.items(),
                             key=lambda x: x[1], reverse=False)
    for word, idx in idx2word_sorted:
        if word == PAD:
            i = np.zeros(model.vector_size)
        elif word == UNK:
            i = np.random.uniform(-.1, .1, model.vector_size)
        else:
            try:
                i = model.wv[word]
            except KeyError:
                i = np.random.uniform(-.1, .1, model.vector_size)
                miss_match += 1
        matrix.append(i)
    matrix = np.stack(matrix, 0)
    # logger.info("Get {}X{} embed_matrix, {:.2f}% use pre-train word2vec".format(
    #     matrix.shape[0], matrix.shape[1], miss_match / matrix.shape[0] * 100
    # ))
    return matrix


if __name__ == "__main__":
    sentences1 = ["我", "我 喜欢 你"]
    sentences2 = ["他"]
    vocab = Vocabulary()
    vocab.save("test")
    vocab.fit(sentences1)
    vocab.transform_one(sentences1[1], max_length=10)
    vocab.training = False
    vocab.fit(sentences2)
    vocab.transform(sentences2, max_length=10)
    vocab.reverse(vocab.transform(sentences2))
    vocab.save("test")
    t=Vocabulary.load("test")
    print(t)

