# Text Classification

有关于文本分类的经典论文、工程实践。

## Paper

+ [Bag of Tricks for Efficient Text Classification, 2016](https://arxiv.org/abs/1607.01759)

论文解读：Tomas Mikolov & Facebook出品，开源项目fastText，简单但在绝大部分文本分类任务中拥有逼近state-of-the-art的性能。1）不使用预训练word2vec，直接利用标签样本进行学习词嵌入矩阵，也许是因为我们最终要通过对词向量做平均得到句向量，所以不追求单个词向量性能？；2）fc隐层输出可作为句向量，供其他任务使用；3）无序N-gram，大幅降低了计算复杂度，但实际性能不比有序N-gram差。

+ [ Convolutional Neural Networks for Sentence Classification, 2014](https://arxiv.org/abs/1408.5882)

论文解读：textCNN，CNN文本分类开山之作，2000引用；1）预训练词向量，情况A：将词向量分为2组，作为不同的channel，一组不fine-tuned，一组fine-tuned，效果较好，情况B：不分组，训练过程中词向量fine-tuned效果也不错；2）不同filter-size，提取不同的N-gram信息；3）单层卷积后，接max-pooling，提取对于任务的最关键信息，并且天然的，max-pooling后向量与句子长度无关，对于变长输入的处理变得简单；4）网络结构：cnn+max-pooling+dropout+fc（l2正则）+softmax。

+ [Pengfei Liu, Xipeng Qiu, Xuanjing Huang, Recurrent Neural Network for Text Classification with Multi-Task Learning,  2016](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)

论文解读：textRNN

+ [Zichao Yang, Diyi Yang , Chris Dyer , Xiaodong He , Alex Smola , Eduard Hovy, Hierarchical Attention Networks for Document Classification, 2016](http://www.aclweb.org/anthology/N16-1174)

论文解读：Attention

+ [Siwei Lai, Liheng Xu, Kang Liu, Jun Zhao, Recurrent Convolutional Neural Networks for Text Classification, 2015](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)

论文解读：textRCNN

## Blog



### reference

[A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/pdf/1404.2188.pdf)

[Paper-attention-layer](https://arxiv.org/pdf/1409.0473v7.pdf)

[Paper-practical-bayesian-optimization-of-machine-learning-algorithms](http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)

[Blog-deeplearning-nlp-best-practices](http://ruder.io/deep-learning-nlp-best-practices/index.html)

[Blog-贝叶斯优化调参之Hyperopt](https://blog.csdn.net/gdh756462786/article/details/79268685)

[知乎-CNN文本分类](https://zhuanlan.zhihu.com/p/28087321)

[知乎-CNN RNN Attention文本分类](https://zhuanlan.zhihu.com/p/25928551)

