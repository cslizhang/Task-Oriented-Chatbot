# coding:utf8
# @Time    : 18-6-7 下午3:34
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from chatbot.utils.log import get_logger
from chatbot.utils.path import MODEL_PATH


class BaseTrainer(object):

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError


class IntentModelTrainer(BaseTrainer):
    def __init__(self, model, lr=0.01, loss=F.cross_entropy, metric=accuracy_score,
                 opt=torch.optim.Adam, start_epoch=0, save_path="default"):
        if save_path == "default":
            self.save_path = MODEL_PATH / "intent"
        else:
            self.save_path = save_path
        self.logger = get_logger(self.__class__.__name__)
        self.model = model
        self.loss = loss
        self.metric = metric
        self.opt = opt(params=self.model.parameters(), lr=lr)
        self.start_epoch = start_epoch

    def evaluate(self, x, y):
        training = self.model.training
        self.model.eval()
        test_x = torch.tensor(x)
        test_y = torch.tensor(y)
        logit = self.model(test_x)
        loss = F.cross_entropy(logit, test_y)
        pred = torch.argmax(logit, 1).detach().numpy()
        acc = self.metric(test_y, pred)
        self.model.training = training
        return {"loss": loss.data, "acc": acc}

    def infer(self, x):
        x = torch.tensor(x)
        logit = self.model(x)
        pred = torch.argmax(logit, 1).numpy()
        return pred

    def _train_step(self, x, y):
        self.opt.zero_grad()
        x = torch.tensor(x)
        y = torch.tensor(y)
        logit = self.model(x)
        loss = self.loss(logit, y)
        loss.backward()
        self.opt.step()
        return loss

    def train(self, train_x, train_y, test_x, test_y,
              epochs, batch_size, log_freq=100, save_best=False):
        self.model.train()
        best_score = 0
        for epoch in range(1, epochs+1):
            for step, (x, y) in enumerate(self._batch_generator(train_x, train_y, batch_size)):
                loss = self._train_step(x, y)
                if step % log_freq == 0:
                    eval_result = self.evaluate(test_x, test_y)
                    self.logger.info(
                        "Epoch: {:>2}, Step: {:>6}, train loss: {:>6.6f}, eval loss: {:>6.6f}, acc: {:>6.6f}.".format(
                            self.start_epoch+epoch, step, loss, eval_result["loss"], eval_result["acc"]
                        ))
                    if best_score < eval_result["acc"]:
                        best_score = eval_result["acc"]
                        if save_best:
                            name = "{}_epoch:{:2>}_step:{}_acc:{:.4f}.pkl".format(
                                    self.model.__class__.__name__,
                                    self.start_epoch + epoch,
                                    step,
                                    eval_result["acc"])
                            self.save(name)
        self.start_epoch += epochs

    @staticmethod
    def _batch_generator(x, y, batch):
        assert x.shape[0] == y.shape[0]
        size = x.shape[0]
        idx = np.array(list(range(0, size)))
        np.random.shuffle(idx)
        x = x[idx].copy()
        y = y[idx].copy()
        n = size // batch
        for i in range(n):
            yield x[batch * i: batch * (i + 1)], y[batch * i: batch * (i + 1)]

    def save(self, name):
        self.save_path.mkdir(exist_ok=True)
        path = str(self.save_path / name)
        torch.save(self.model.state_dict(), path)
