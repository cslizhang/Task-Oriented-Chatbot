# coding:utf8
# @Time    : 18-5-15 下午4:16
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# sys.path.append("~/project/Tast-Oriented-Chatbot")
from chatbot.utils.path import root, MODEL_PATH
from chatbot.utils.log import get_logger
from chatbot.preprocessing import text


logger = get_logger("TextCNN")


class FastText(nn.Module):
    def __init__(self, param: dict):
        super().__init__()
        embed_dim = param['embed_dim']
        vocab_size = param['vocab_size']
        class_num = param['class_num']
        hidden_size = param["hidden_size"]
        self.param = param
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear()


class TextCNN(nn.Module):
    def __init__(self, param: dict):
        super().__init__()
        ci = 1  # input chanel size
        kernel_num = param['kernel_num'] # output chanel size
        kernel_size = param['kernel_size']
        vocab_size = param['vocab_size']
        embed_dim = param['embed_dim']
        dropout = param['dropout']
        class_num = param['class_num']
        self.param = param
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_size[0], embed_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], embed_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, class_num)

    def init_embed(self, embed_matrix):
        self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length, embed_dim)
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        # x: (batch, sentence_length)
        x = self.embed(x)
        # x: (batch, sentence_length, embed_dim)
        # TODO init embed matrix with pre-trained
        x = x.unsqueeze(1)
        # x: (batch, 1, sentence_length, embed_dim)
        x1 = self.conv_and_pool(x, self.conv11)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv12)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv13)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        logit = F.log_softmax(self.fc1(x), dim=1)
        # logit = F.softmax(self.fc1(x), dim=1)
        # logit = self.fc1(x)
        return logit


def save_state_dict(model, epoch, step, eval_acc):
    save_path = MODEL_PATH / "{}_epoch_{}_step_{}_acc_{:.4f}.params.pkl".format(
            model._get_name(), epoch, step, eval_acc
        )
    save_path = save_path.resolve()
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), str(save_path))
    logger.info("save model in {}".format(str(save_path)))


def batch_generator(x, y, batch):
    assert x.shape[0] == y.shape[0]
    size = x.shape[0]
    idx = np.array(list(range(0, size)))
    np.random.shuffle(idx)
    x = x[idx].copy()
    y = y[idx].copy()
    n = size // batch
    for i in range(n):
        yield x[batch*i: batch*(i+1)], y[batch*i: batch*(i+1)]


def train(model, train_x, train_y, test_x, test_y,
          lr, epochs, init_epochs, batch_size):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    best_score = 0
    for epoch in range(init_epochs, epochs):
        for step, (x, y) in enumerate(batch_generator(train_x, train_y, batch_size)):
            x = torch.tensor(x)
            y = torch.tensor(y)
            optimizer.zero_grad()
            logit = model(x)
            loss = F.cross_entropy(logit, y)
            # loss = F.nll_loss(logit, y)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                eval_loss, eval_acc = eval(model, test_x, test_y)
                logger.info("Epoch: {:>2}, Step: {:>6}, train loss: {:>6.6f}, eval loss: {:>6.6f}, acc: {:>6.6f}.".format(
                    epoch, step, loss, eval_loss, eval_acc
                ))
                if best_score < eval_acc:
                    best_score = eval_acc
                    if eval_acc >= 0.97:
                        save_state_dict(model, epoch, step, eval_acc)


def eval(model, test_x, test_y):
    model.eval()
    test_x = torch.tensor(test_x)
    test_y = torch.tensor(test_y)
    logit = model(test_x)
    loss = F.cross_entropy(logit, test_y)
    pred = torch.argmax(logit, 1).detach().numpy()
    acc = accuracy_score(test_y, pred)
    return loss, acc


def predict(sentence, model, vocabulary, label):
    s = text.cut(sentence)
    s = torch.tensor(vocabulary.transform(s, 10))
    pred = torch.argmax(model(s), 1)
    print(pred)
    pred_class = label.idx2word[pred.data.item()]
    return pred_class


if __name__ == "__main__":
    from chatbot.cparse.text import read_fasttext_file
    from chatbot.cparse.vocabulary import Vocabulary
    from chatbot.cparse.label import Label
    p = root.parent / "corpus" / "intent" / "fastText"
    train_x, train_y = read_fasttext_file(str(p/"demo.train.txt"))
    test_x, test_y = read_fasttext_file(str(p / "demo.train.txt"))
    vocab = Vocabulary()
    vocab.update(train_x)
    label = Label()
    label.update(train_y)
    train_x = np.array(vocab.transform(train_x, max_length=10))
    test_x = np.array(vocab.transform(test_x, max_length=10))
    train_y = np.array(label.transform(train_y))
    test_y = np.array(label.transform(test_y))

    textCNN_param = {
        "vocab_size": len(vocab),
        "embed_dim": 60,
        "class_num": len(label),
        "kernel_num": 16,
        "kernel_size": [3, 4, 5],
        "dropout": 0.5,
    }
    model = TextCNN(textCNN_param)
    new_model = TextCNN(textCNN_param)
    new_model.load_state_dict(torch.load("/home/zhouzr/project/Task-Oriented-Chatbot/chatbot/results/intent/TextCNN_epoch_4_step_100_acc_0.9881.params.pkl"))
    train(model, train_x, train_y, test_x, test_y, 0.01, 4, 0, 100)
    s = ["你真是可爱阿", "你很喜欢学习哦", "我再也不想理你了",
         "吃饭没", "明天会下雨马", "你哥哥是谁", "你有哥哥么", "弟弟是谁",
         "我想买手机", "我是你主人"
         ]
    for i in s:
        print(i, predict(i, model, vocab, label))
