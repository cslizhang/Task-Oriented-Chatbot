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

    p = ROOT_PATH.parent / "corpus" / "intent" / "fastText"
    train_x, train_y = read_fasttext_file(str(p / "demo.train.txt"))
    test_x, test_y = read_fasttext_file(str(p / "demo.train.txt"))
    vocab = Vocabulary()
    vocab.fit(train_x)
    label = IntentLabel()
    label.fit(train_y)
    train_x = np.array(vocab.transform(train_x, max_length=10))
    test_x = np.array(vocab.transform(test_x, max_length=10))
    train_y = np.array(label.transform(train_y))
    test_y = np.array(label.transform(test_y))

    fasttext_param = {
        "vocab_size": len(vocab),
        "embed_dim": 60,
        "class_num": len(label),
        "lr": 0.01,
        # "dropout": 0.5,
    }
    model = FastText(fasttext_param)
    model.fit(train_x, train_y, test_x, test_y, 1, 64, save_best=False)
    # model.save("test")
    # x = FastText.load(str(MODEL_PATH / "intent" / "test.FastText"))
    s = ["你真是可爱阿", "你很喜欢学习哦", "我再也不想理你了",
         "吃饭没", "明天会下雨马", "你哥哥是谁", "你有哥哥么", "弟弟是谁",
         "我想买手机", "我是你主人"
         ]
    from chatbot.preprocessing.text import cut
    for i in s:
        print(i, label.reverse_one(model.infer(np.array(vocab.transform_one(cut(i), max_length=10)))[0]))

