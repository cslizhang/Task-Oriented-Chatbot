# -*- coding: utf-8 -*-
# @Time    : 5/12/18 13:23
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.utils import path
from chatbot.utils.log import get_logger

from pathlib import Path
import multiprocessing as mp

import jieba

CPU = mp.cpu_count()
logger = get_logger("Text cut")
SEG_VOCAB_PATH = Path(path.ROOT_PATH, "config", "vocab_jieba_seg").resolve().absolute()
jieba.load_userdict(str(SEG_VOCAB_PATH))
jieba.initialize()


def _cut2list(x):
    return list(jieba.cut(x))


def _cut2str(x):
    return " ".join(jieba.cut(x))


# @time_counter
def cut(x, n_job=None, join=False):
    assert isinstance(x, str) or isinstance(x, list)
    if isinstance(x, str):
        # logger.info("String input, user 1 cpu core")
        if join:
            return _cut2str(x)
        else:
            return _cut2list(x)
    if n_job:
        n_job = min(CPU, n_job)
        # logger.info("%d Sentences input, Use %d cpu core " % (len(x), n_job))
        pool = mp.Pool(n_job)
        if join:
            rst = pool.map(_cut2str, x)
        else:
            rst = pool.map(_cut2list, x)
        pool.close()
        pool.join()
    else:
        if join:
            rst = [_cut2str(i) for i in x]
        else:
            rst = [_cut2list(i) for i in x]
    return rst


if __name__ == "__main__":
    texts = ["我很好才怪\n", "市场化交易下的售电公司如何发展？"] * 20000
    print(cut(texts, n_job=1))
    print(cut(texts[0], n_job=4, join=True))
    # cut(texts)
