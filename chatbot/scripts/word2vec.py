# -*- coding: utf-8 -*-
# @Time    : 5/13/18 09:50
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
# import gensim
#
# sentence=[["我", "很喜欢", "上学"], ["不" ,"上学", "真的", "好吗"]]
# model = gensim.models.Word2Vec(sentence, size=10, window=5, min_count=1)
# model.wv["很喜欢"]
# model.most_similar("我")
# model.save("~/project/model")
#
# m = gensim.models.Word2Vec.load("~/project/model")
# m.wv["我"]
# model.build_vocab([["他", "很喜欢", "我"]], update=True)
# model.train([["他", "很喜欢", "上学"],], total_examples=2, epochs=10)
# model.wv["很喜欢"]
# model.most_similar("他")
# x=model.wv.vocab["我"]
# model.wv(["我", "很喜欢"])
