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

    def transform(self, x, max_length=None):
        """文本转换成id

        :param x: <list of list> 最小元素为string,代表一个单词
        :param max_length: 固定返回的每个句子固定长度
        :return: <list of list> 最小元素为单词的idx
        """
        rst = []
        for sentence in x:
            rst_s = self.transform_one(sentence)
        rst.append(rst_s)
        return rst

    def transform_one(self, x, max_length=None):
        rst_s = [self.word2idx.get(word, UNK_IDX) for word in x]
        if max_length is not None:
            rst_s = rst_s[:max_length]
            rst_s = [PAD_IDX] * (max_length - len(rst_s)) + rst_s
        return rst_s


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

