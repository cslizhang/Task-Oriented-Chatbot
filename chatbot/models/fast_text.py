# coding:utf8
# @Time    : 18-5-15 下午4:16
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from pathlib import Path
from chatbot.utils.path import root

from fastText import train_supervised, load_model


#训练模型
train_path = root.parent / "corpus" / "intent" / "fastText" / "demo.train.txt"
test_path = root.parent / "corpus" / "intent" / "fastText" / "demo.test.txt"
model_path = root.parent / "chatbot" / "results" / "fasttext"
model_path.parent.exists()
model = train_supervised(str(train_path),label="__label__",
                            epoch=30, lr=.1, wordNgrams=2)

model.save_model(str(model_path))

#load训练好的模型
model = load_model(str(model_path))
result = model.test(str(test_path))
