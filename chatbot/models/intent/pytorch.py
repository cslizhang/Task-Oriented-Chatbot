# -*- coding: utf-8 -*-
# @Time    : 5/22/18 14:00
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score

# sys.path.append("~/project/Tast-Oriented-Chatbot")
from chatbot.utils.path import root, MODEL_PATH
from chatbot.utils.log import get_logger
from chatbot.preprocessing import text


logger = get_logger("Intent Model Train")


def save_state_dict(self, epoch, step, eval_acc):
    save_path = MODEL_PATH / "{}_epoch_{}_step_{}_acc_{:.4f}.params.pkl".format(
        self._get_name(), epoch, step, eval_acc
        )
    save_path = save_path.resolve()
    save_path.parent.mkdir(exist_ok=True)
    torch.save(self.state_dict(), str(save_path))
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
          lr, epochs, init_epochs, batch_size, save_best=False):
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
                eval_loss, eval_acc = evaluate(model, test_x, test_y)
                logger.info("Epoch: {:>2}, Step: {:>6}, train loss: {:>6.6f}, eval loss: {:>6.6f}, acc: {:>6.6f}.".format(
                    epoch, step, loss, eval_loss, eval_acc
                ))
                if best_score < eval_acc:
                    best_score = eval_acc
                    if save_best:
                        model.save_state_dict(epoch, step, eval_acc)


def evaluate(model, test_x, test_y):
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
    pred_class = label.idx2word[pred.data.item()]
    return pred_class

