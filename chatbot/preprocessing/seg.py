# -*- coding: utf-8 -*-
# @Time    : 5/12/18 13:23
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.utils import path
from chatbot.utils.log import get_logger
from chatbot.utils.wrapper import time_counter

from pathlib import Path
import multiprocessing as mp
import concurrent.futures

import jieba
jieba.initialize()


logger = get_logger(__file__)
SEG_VOCAB_PATH = Path(path.root, "data", "seg_vocab").resolve().absolute()
jieba.load_userdict(str(SEG_VOCAB_PATH))


@time_counter
def cut(x, n_job=None):
    def _cut(x):
        return [" ".join(jieba.cut(i)) for i in x]
    assert isinstance(x, str) or isinstance(x, list)
    if isinstance(x, str):
        logger.debug("cut function only support list, but str input")
        x = [x]
    if n_job:
        n_job = max(mp.cpu_count(), n_job)
        pool = mp.Pool(n_job)
        rst = pool.map(_cut, x)
        pool.close()
        pool.join()
    else:
        rst = _cut(x)
    return rst


if __name__ == "__main__":
    texts = ["我很好才怪", "市场化交易下的售电公司如何发展？"] * 20000
    cut(texts, n_job=4)
    # cut(texts)
