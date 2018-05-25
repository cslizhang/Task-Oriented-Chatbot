# coding:utf8
# @Time    : 18-5-15 下午4:16
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score


class FastText(nn.Module):
    def __init__(self, param: dict):
        super().__init__()
        embed_dim = param['embed_dim']
        vocab_size = param['vocab_size']
        class_num = param['class_num']
        # hidden_size = param["hidden_size"]
        self.param = param
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, class_num)

    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim=1, keepdim=False)
        output = self.fc(x)
        output = F.log_softmax(output, dim=1)
        return output


if __name__ == "__main__":
    import sys
    from pathlib import Path
    p = str(Path(".", "..", "..", "..").resolve())
    sys.path.append(p)
    from chatbot.cparse.text import read_fasttext_file
    from chatbot.cparse.vocabulary import Vocabulary
    from chatbot.cparse.label import Label
    from chatbot.models.intent.pytorch import *

    p = root.parent / "corpus" / "intent" / "fastText"
    train_x, train_y = read_fasttext_file(str(p / "demo.train.txt"))
    test_x, test_y = read_fasttext_file(str(p / "demo.train.txt"))
    vocab = Vocabulary()
    vocab.update(train_x)
    label = Label()
    label.update(train_y)
    train_x = np.array(vocab.transform(train_x, max_length=10))
    test_x = np.array(vocab.transform(test_x, max_length=10))
    train_y = np.array(label.transform(train_y))
    test_y = np.array(label.transform(test_y))

    fasttext_param = {
        "vocab_size": len(vocab),
        "embed_dim": 60,
        "class_num": len(label),
        # "dropout": 0.5,
    }
    model = FastText(fasttext_param)
    # new_model.load_state_dict(torch.load("/home/zhouzr/project/Task-Oriented-Chatbot/chatbot/results/intent/TextCNN_epoch_4_step_100_acc_0.9881.params.pkl"))
    train(model, train_x, train_y, test_x, test_y, 0.01, 4, 0, 100)
    s = ["你真是可爱阿", "你很喜欢学习哦", "我再也不想理你了",
         "吃饭没", "明天会下雨马", "你哥哥是谁", "你有哥哥么", "弟弟是谁",
         "我想买手机", "我是你主人"
         ]
    for i in s:
        print(i, predict(i, model, vocab, label))