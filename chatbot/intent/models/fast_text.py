# coding:utf8
# @Time    : 18-5-15 下午4:16
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import torch
from torch import nn
from torch.nn import functional as F
from chatbot.intent.models.base_intent_model import BaseIntentModel


class FastText(BaseIntentModel):
    def __init__(self, param: dict):
        super().__init__(param)
        embed_dim = param['embed_dim']
        vocab_size = param['vocab_size']
        class_num = param['class_num']
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, class_num)

    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim=1, keepdim=False)
        output = self.fc(x)
        output = F.log_softmax(output, dim=1)
        return output


if __name__ == "__main__":
    import numpy as np
    from chatbot.utils.path import ROOT_PATH, MODEL_PATH
    from chatbot.utils.data import read_fasttext_file
    from chatbot.cparse.vocabulary import Vocabulary
    from chatbot.cparse.label import IntentLabel

    # p = ROOT_PATH.parent / "corpus" / "intent" / "fastText"
    # train_x, train_y = read_fasttext_file(str(p / "demo.train.txt"))
    # test_x, test_y = read_fasttext_file(str(p / "demo.train.txt"))
    # x, y = read_fasttext_file(str(p / "amazon.txt"))
    # train_x, train_y = x[:7000], y[:7000]
    # test_x, test_y = x[7000:], y[7000:]
    import pandas as pd
    from chatbot.preprocessing.text import cut
    corpus = pd.read_excel(ROOT_PATH.parent/"corpus"/"intent"/"intent_corpus.xlsx")
    x = cut(corpus.text.tolist())
    y = corpus.intent.tolist()
    vocab = Vocabulary()
    vocab.fit(x)
    label = IntentLabel()
    label.init_from_config()
    # label.fit(y)
    train_x = np.array(vocab.transform(x, max_length=10))
    test_x = np.array(vocab.transform(x, max_length=10))
    train_y = np.array(label.transform(y))
    test_y = np.array(label.transform(y))

    fasttext_param = {
        "vocab_size": len(vocab),
        "embed_dim": 60,
        "class_num": len(label),
        "lr": 0.01,
        # "dropout": 0.5,
    }
    model = FastText(fasttext_param)
    model.fit(train_x, train_y, test_x, test_y, 2, 8, save_best=False)

    def test(s):
        s = vocab.transform_sentence(cut(s), max_length=10)
        return label.reverse_one(model.infer(s)[0])
    test("上个月最高用电")

    model.save(str(MODEL_PATH/"v0.2"/"intent_model"))
    vocab.save(str(MODEL_PATH/"v0.2"/"vocab"))
    label.save(str(MODEL_PATH / "v0.2" / "label"))
