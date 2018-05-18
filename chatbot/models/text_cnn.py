# coding:utf8
# @Time    : 18-5-15 下午4:16
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import torch
from torch import nn
from torch import autograd
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
#
# {vocab_size, embed_dim, class_num,
#                  kernel_num, kernel_sizes, dropout,
#                  fine_tuned, use_pre_trained_word2vec=None}


class TextCNN(nn.Module):
    def __init__(self, param: dict):
        super().__init__()
        ci = 1  # input chanel size
        co = param['kernel_num'] # output chanel size
        kernel_size = param['kernel_size']
        vocab_size = param['vocab_size']
        embed_dim = param['embed_dim']
        dropout = param['dropout']
        class_num = param['class_num']
        self.param = param
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv13 = nn.Conv2d(ci, co, (3, embed_dim))
        self.conv14 = nn.Conv2d(ci, co, (4, embed_dim))
        self.conv15 = nn.Conv2d(ci, co, (5, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size) * co, class_num)

    def conv_and_pool(self, x, conv):
        #
        x = F.relu(conv(x).squeeze(3))
        # remove dimension 3 (batch, chanel_out, sentence_length)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, chanel_out)
        return x

    def forward(self, x):
        x = self.embed(x)  # (batch, sentence_length, embed_dim)
        # TODO init embed matrix with pre-trained
        x = x.unsqueeze(1) # (batch, ci, sentence_length, embed_dim)
        x1 = self.conv_and_pool(x, self.conv13)
        x2 = self.conv_and_pool(x, self.conv14)
        x3 = self.conv_and_pool(x, self.conv15)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3))  # (batch, len(kernel_size) * kernel_num)
        x = self.dropout(x)
        logit = self.fc1(x)
        # logit = F.log_softmax(x)
        return logit


def save(model, path, save_prefix, steps):
    pass


class TextClassificationTestDataSet(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


