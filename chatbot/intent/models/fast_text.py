# coding:utf8
# @Time    : 18-5-15 下午4:16
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from torch import nn


class FastText(nn.Module):
    def __init__(self, param: dict):
        super().__init__()
        embed_dim = param['embed_dim']
        vocab_size = param['vocab_size']
        class_num = param['class_num']
        # hidden_size = param["hidden_size"]
        self.param = param
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, class_num)

    def forward(self, x):
        x = self.embed(x)
        x = torch.mean(x, dim=1, keepdim=False)
        output = self.fc(x)
        output = F.log_softmax(output, dim=1)
        return output


if __name__ == "__main__":
    from chatbot.core.trainer import IntentModelTrainer
    import sys
    from pathlib import Path
    p = str(Path(".", "..", "..", "..").resolve())
    sys.path.append(p)
    from chatbot.utils.data import read_fasttext_file
    from chatbot.cparse.vocabulary import Vocabulary
    from chatbot.cparse.label import Label
    from chatbot.intent.models.pytorch import *



    p = root.parent / "corpus" / "intent" / "fastText"
    train_x, train_y = read_fasttext_file(str(p / "demo.train.txt"))
    test_x, test_y = read_fasttext_file(str(p / "demo.train.txt"))
    vocab = Vocabulary()
    vocab.update(train_x)
    label = Label()
    label.update(train_y)
    train_x = np.array(vocab.transform(train_x, max_length=10))
    test_x = np.array(vocab.transform(test_x, max_length=10))
    train_y = np.array(label.transform(train_y))
    test_y = np.array(label.transform(test_y))

    fasttext_param = {
        "vocab_size": len(vocab),
        "embed_dim": 60,
        "class_num": len(label),
        # "dropout": 0.5,
    }
    model = FastText(fasttext_param)
    trainer = IntentModelTrainer(model=model)
    trainer.train(train_x, train_y, test_x, test_y, 4, 64, save_best=True)
