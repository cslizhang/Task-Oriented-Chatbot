# -*- coding: utf-8 -*-
# @Time    : 5/12/18 13:23
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.utils import path
from chatbot.utils.log import get_logger
from chatbot.utils.wrapper import time_counter

from pathlib import Path
import multiprocessing as mp
# import concurrent.futures

import jieba


logger = get_logger("Text cut")
SEG_VOCAB_PATH = Path(path.root, "config", "vocab_jieba_seg").resolve().absolute()
jieba.load_userdict(str(SEG_VOCAB_PATH))
jieba.initialize()


def _cut(x):
    return list(jieba.cut(x))


@time_counter
def cut(x, n_job=None):
    assert isinstance(x, str) or isinstance(x, list)
    if isinstance(x, str):
        logger.info("String input, user 1 cpu core")
        return _cut(x)
    if n_job:
        n_job = max(2, n_job)
        logger.info("%d Sentences input, Use %d cpu core " % (len(x), n_job))
        pool = mp.Pool(n_job)
        rst = pool.map(_cut, x)
        pool.close()
        pool.join()
    else:
        rst = [_cut(i) for i in x]
    return rst


if __name__ == "__main__":
    texts = ["我很好才怪\n", "市场化交易下的售电公司如何发展？"] * 1000
    print(cut(texts, n_job=2))
    print(cut(texts[1], n_job=2))
    # cut(texts)
