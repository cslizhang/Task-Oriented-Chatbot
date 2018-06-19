# coding:utf8
# @Time    : 18-6-19 上午11:05
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import torch
from torch import nn
from torch.nn import functional as F
from chatbot.intent.models.base_intent_model import BaseIntentModel


class AttFastText(BaseIntentModel):
    def __init__(self, param: dict):
        super().__init__(param)
        embed_dim = param['embed_dim']
        vocab_size = param['vocab_size']
        class_num = param['class_num']
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, class_num)
        self.score_w = nn.Parameter(torch.randn(param['embed_dim']))
        self.score_b = nn.Parameter(torch.randn(1))

    def _score(self, x, mask=None):
        # inputs: embedded x
        score = F.tanh(torch.matmul(x, self.score_w) + self.score_b)
        if mask:
            score.data.masked_fill_(mask, -float('inf'))
        score = F.softmax(score, 1)
        return score

    def forward(self, x):
        x = self.embed(x)
        score = self._score(x)
        print(score[0])
        # x = torch.mean(x, dim=1, keepdim=False)
        x_att = torch.sum(score.unsqueeze(2).expand_as(x) * x, dim=1, keepdim=False)
        output = self.fc(x_att)
        output = F.log_softmax(output, dim=1)
        return output

if __name__ == "__main__":
    import numpy as np
    from chatbot.utils.path import ROOT_PATH, MODEL_PATH
    from chatbot.utils.data import read_fasttext_file
    from chatbot.cparse.vocabulary import Vocabulary
    from chatbot.cparse.label import IntentLabel

    p = ROOT_PATH.parent / "corpus" / "intent" / "fastText"
    x, y = read_fasttext_file(str(p / "corpus"))
    train_x, train_y = x[:7000], y[:7000]
    test_x, test_y = x[7000:], y[7000:]
    import copy
    x = copy.deepcopy(train_x)
    # test_x, test_y = read_fasttext_file(str(p / "demo.train.txt"))
    vocab = Vocabulary()
    vocab.fit(train_x)
    label = IntentLabel()
    label.fit(train_y)
    train_x = np.array(vocab.transform(train_x, max_length=10))
    test_x = np.array(vocab.transform(test_x, max_length=10))
    train_y = np.array(label.transform(train_y))
    test_y = np.array(label.transform(test_y))

    p = {
        "vocab_size": len(vocab),
        "embed_dim": 60,
        "class_num": len(label),
        "lr": 0.01,
        # "dropout": 0.5,
    }
    model = AttFastText(p)
    model.fit(train_x, train_y, test_x, test_y, 2, 32, save_best=False)
    # model.param["lr"] = 0.003
    # model.fit(train_x, train_y, test_x, test_y, 4, 64, save_best=False)
    # model.save("test")
    # x = FastText.load(str(MODEL_PATH / "intent" / "test.FastText"))
    s = ["你真是可爱阿", "你很喜欢学习哦", "我再也不想理你了",
         "吃饭没", "明天会下雨马", "你哥哥是谁", "你有哥哥么", "弟弟是谁",
         "我想买手机", "我是你主人", "我可以给你打分吗，评价"
         ]
    from chatbot.preprocessing.text import cut

    for i in s:
        print(i, label.reverse_one(model.infer(np.array(vocab.transform_one(cut(i), max_length=10)))[0]))
    from chatbot.evaluate.plot import plot_attention_1d
    idx=1200
    att = model._score(torch.tensor(np.array(vocab.transform_one(train_x[idx], max_length=10)).reshape(-1, 10)))
    print(label.reverse_one(model.infer(train_x[idx])[0]))
    plot_attention_1d([vocab.reverse_one(train_x[idx]).split(" ")],
                      att.detach().numpy())
