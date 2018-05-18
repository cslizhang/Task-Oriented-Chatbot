# coding:utf8
# @Time    : 18-5-15 下午4:16
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import sys

import torch
from torch import nn
from torch import autograd
from torch.nn import functional as F
from torch.utils.data import Dataset
from chatbot.utils.log import get_logger


logger = get_logger("TextCNN")


textCNN_param = {
    "vocab_size": 1000,
    "embed_dim": 60,
    "class_num": 10,
    "kernel_num": 2,
    "kernel_size": [3, 4, 5],
    "dropout": 0.5,
}


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
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_size[0], embed_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], embed_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, class_num)

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
        x1 = self.conv_and_pool(x, self.conv13)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv14)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv15)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        logit = self.fc1(x)
        # logit = F.log_softmax(x)
        return logit


def save_state_dict(model, path, model_name, steps, loss):
    from pathlib import Path
    file_name = "{}_statedict_steps_{}_loss_{:.6f}.pt".format(
        model_name, steps, loss)
    path = Path(path).resolve() / file_name
    path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)


def train(model, train_loader, eval_loader,
          lr, epoch, init_epoch=0,
          use_cuda=False, log_interval=100,
          eval_interval=100, save_interval=100
          ):
    logger.info("Start train %s" % model.__name__)
    if use_cuda:
        model.cuda()
        logger.info("Use CUDA ..")
    else:
        logger.info("Use CPU ..")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps = 0
    model.train()
    for epoch in range(1+init_epoch, epoch+1+init_epoch):
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            logit = model(x)
            loss = F.cross_entropy(logit, y)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % log_interval == 0:
                logger.info("Train Epoch {} [{} / {} ({:.1f}%)]\nLoss {:.6f}".format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()
                ))
            if steps % eval_interval ==0:
                eval_loss = eval()
                logger.info("Eval Epoch {}, Step {}, loss {:.6f}".format(
                    epoch, steps, loss.item()
                ))
            if steps % save_interval ==0:
                pass


def eval(model, eval_loader):
    model.eval()
    loss = 0
    for (x, y) in eval_loader:
        logit = model(x)
        loss_batch = F.cross_entropy(logit, y, size_average=True)
        loss += loss_batch.item()


def predict():
    pass


class TextClassificationTestDataSet(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


