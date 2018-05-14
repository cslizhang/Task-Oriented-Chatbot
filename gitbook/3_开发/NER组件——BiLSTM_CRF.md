#### NER组件—BiLSTM_CRF

pytorch有一个[官方的BiLSTM_CRF的实现](http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)，但由于主要用于演示说明，在工程上应用还有一些障碍，本文以官方实现为基础进行代码梳理、重构。

阅读本文需要掌握的知识要点：

+ 双向LSTM原理及实现
+ 维特比算法（已知HMM模型参数，寻找最可能的能产生某一特定输出序列的隐含状态的序列）

##### 官方代码解读

```python

句子级别的对数似然，其实就是考虑到CRF模型在序列标注问题中的优势，
将标签转移得分加入到了目标函数中，这个思想称为结合了一层CRF层。

它主要有Embedding层、双向RNN层、tanh隐层以及最后的CRF层构成。
它与之前NN/CNN-CRF的主要区别就是他使用的是双向RNN代替了NN/CNN。
实验结果表明RNN-CRF获得了更好的效果，已经达到或者超过了基于丰富特征的CRF模型，
成为目前基于深度学习的NER方法中的最主流模型。
在特征方面，该模型继承了深度学习方法的优势，无需特征工程，
使用词向量以及字符向量就可以达到很好的效果，
如果有高质量的词典特征，能够进一步获得提高。


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

def to_scalar(var):
    # 将shape=[1] Tensor转标量
    assert var.shape == torch.Size([1])
    return var.data.tolist()[0]

def argmax(vec):
    # 返回一维Tensor的最大值索引
    assert vec.shape[0] == 1
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def prepare_sequence(seq, to_ix):
    # 单词序列到id序列映射，to_ix为外部传入的映射字典
    idxs = torch.LongTensor([to_ix[w] for w in seq])
    return autograd.Variable(idxs)

def log_sum_exp(vec):
    return vec.max() + torch.log(torch.sum(torch.exp(vec - vec.max())))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        # lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # 有点没明白为什么LSTM的h0 c0不用全零,而用标准高斯初始化?
        lstm_out, self.hidden = self.lstm(embeds)
        # lstm.shape = [seq_len, batch_size=1, hidden * n_layers]
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        # 这步可以在fc层后面做
        # lstm.shape = [seq_len, hidden * n_layers]
        lstm_feats = self.hidden2tag(lstm_out).view(len(sentence), -1)
        # 这步可以在lstm_out做reshape
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        # score 初始化为0
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]])])
        # 将开始\结束标签拼接在tags首尾
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        # 由于采用双向rnn,输出序列是原始序列的倒序
        # score(t) = score(t+1) + 第t+1到第t的转移概率 + 第t+1的概率
        # score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        # 已经在前面的代码加了
        # 最后返回路径总分
        return score

    def _viterbi_decode(self, feats):
        # 维特比解码,将bi-LSTM输出结果解码成最佳tags序列
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        # 动态规划思想
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
        # 对于lstm的倒序的每一步
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # 对于每一种可能状态
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                # 计算转移分数
                best_tag_id = argmax(next_tag_var)
                # 得到最高的得分tag
                bptrs_t.append(best_tag_id)
                # 存储最高得分tag
                viterbivars_t.append(next_tag_var[0][best_tag_id])
                # 记录最高概率路径
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
```
