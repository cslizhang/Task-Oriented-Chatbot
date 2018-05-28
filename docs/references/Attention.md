# Attention

对于序列建模，假设有句子矩阵$X=(x_1, x_2, … , x_n)$，其中$x_i$代表第$i$个词的词向量，维度为$d$，故$X \in R^{n\times d}$，对$X$的建模思路有下面3种：

+ RNN： $y_i = f(y_{i-1}, x_i)$
+ CNN：$y_i = f(x_{i-1}, x_{i}, x_{i+1})$
+ Attention：$y_i = f(x_i, A, B)$

首先给出Attention的定义，$Attention(Q,K,V)=softmax(\frac{nonlinear(Q,K)}{\sqrt[2]{d_k}})V$

其中$Q\in R^{n\times d_k}$，代表Query，通常是decoder的隐藏状态，$K \in R^{m\times d_k}$，代表Key，通常是encoder的隐藏状态，$V \in R^{m \times d_v}$，代表与Key一一对应的标准化权重Value，代表着Key

The [query] is usually the hidden state of the decoder. A [key] is the hidden state of the encoder and the corresponding [value] is the normalized weight, representing to what extent the key is being attended to. The way the query and the key are combined to get a value is by a weighted sum. Here, a dot product of the query and a key is taken to get a value.





#### 引用

[苏剑林[《Attention is All You Need》浅读（简介+代码）](https://kexue.fm/archives/4765)](https://kexue.fm/archives/4765)

