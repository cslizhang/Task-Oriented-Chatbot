# -*- coding: utf-8 -*-
# @Time    : 5/26/18 15:32
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import torch
from torch import nn
import torch.nn.functional as F


class CRAN(nn.Module):
    """CNN-based Attention Rnn network"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        vocab_size = args["vocab_size"]
        embed_dim = args["embed_dim"]
        rnn_dim = args["rnn_dim"]
        class_num = args["class_num"]
        kernel_num = args['kernel_num']  # output chanel size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, rnn_dim, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(rnn_dim, class_num)
        self.conv = nn.Conv2d(1, kernel_num, (3, embed_dim), padding=(1, 0))
        self.dropout = nn.Dropout(args["dropout"])

    def init_hidden(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.args["rnn_dim"])
        c0 = torch.zeros(1, batch_size, self.args["rnn_dim"])
        return h0, c0


    def forward(self, x):
        # x (batch, sentence)
        x_embed = self.embed(x)
        # x_embed (batch, sentence, embed_dim)
        # forward of attention branch
        x_conv = self.conv(x_embed.unsqueeze(1)).squeeze(2)
        # x_conv (batch, out_chanel, sentence)
        att = F.softmax(torch.mean(x_conv, 1, False), dim=1)
        # x_att (batch, sentence)
        return att

        # hidden = self.init_hidden(x_embed.size()[0])
        # lstm_out, (h_n, c_n) = self.lstm(x_embed, hidden)
        # lstm_out (batch, sentence, encode_dim)


if __name__ == "__main__":
    param = {
        "vocab_size": 1000,
        "embed_dim": 60,
        "rnn_dim":30,
        "dropout":0.5,
        "class_num":10,
        "kernel_num":4
    }
    rnn = CRAN(param)
    test_input = torch.arange(0, 200).view(10, 20).long()
    out = rnn(test_input)
