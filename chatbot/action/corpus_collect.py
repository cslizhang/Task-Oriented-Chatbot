# coding:utf8
# @Time    : 18-6-2 下午2:00
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from chatbot.core.action import BaseAction


class IntentCorpusCollect(BaseAction):
    def __init__(self):
        super().__init__(name="IntentCollect", idx=1)

    def __call__(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    act1 = IntentCorpusCollect()
    print(act1)
