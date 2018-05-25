# coding:utf8
# @Time    : 18-5-21 上午11:46
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import numpy as np

from chatbot.cparse.constant import *
from chatbot.utils.log import get_logger
from chatbot.cparse.base import Dictionary


logger = get_logger(__name__)


class Vocabulary(Dictionary):
    def __init__(self):
        super().__init__()
        self.word2idx = {PAD: PAD_IDX, UNK: UNK_IDX}
        self.idx = len(ALL_CONSTANT)

    def transform(self, x, max_length=None):

        if isinstance(x, str):
            return self.word2idx.get(x, UNK_IDX)
        elif isinstance(x, list) and isinstance(x[0], str):
            ri = [self.word2idx.get(w, UNK_IDX) for w in x]
            if max_length is not None:
                ri = ri[:max_length]
                ri = [PAD_IDX] * (max_length - len(ri)) + ri
            result = [ri]
            return result
        else:
            result = []
            for s in x:
                ri = [self.word2idx.get(w, UNK_IDX) for w in s]
                if max_length is not None:
                    ri = ri[:max_length]
                    ri = [PAD_IDX] * (max_length - len(ri)) + ri
                result.append(ri)
            return result


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
    logger.info("Get {}X{} embed_matrix, {:.2f}% use pre-train word2vec".format(
        matrix.shape[0], matrix.shape[1], miss_match / matrix.shape[0] * 100
    ))
    return matrix


if __name__ == "__main__":
    s1 = [["我", "吃"], ["吃", "什么"]]
    s2 = [["我", "haha"], ["吃", "什么"]]
    d = Vocabulary()
    d.update(s1)
    d.update(s2)
    d.transform(s1, max_length=10)
    d.save("/home/zhouzr/dict")
