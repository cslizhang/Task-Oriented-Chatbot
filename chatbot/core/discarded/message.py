# coding:utf8
# @Time    : 18-6-2 下午3:12
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import datetime as dt
from chatbot.core.discarded.user import User


class BaseMessage(object):
    def __init__(self, inputs):
        """消息初始化

        :param inputs: <dict>
        """
        self.text = inputs["text"]
        self.user = User(inputs["user"], inputs["jurisdiction"])
        self.interface = inputs["interface"]
        self.time = dt.datetime.now()

    @property
    def type(self):
        raise NotImplementedError

    def __str__(self):
        return "A {} Message <time: {}> <text: {}>".format(
            self.type,
            str(self.time),
            self.text
        )

    def __repr__(self):
        return self.__str__()

    # TODO: 消息检查
    def check_state(self):
        pass


class Response(BaseMessage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def type(self):
        return "Response"


class Query(BaseMessage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def type(self):
        return "Query"


if __name__ == "__main__":
    test_inputs1 = {
        "text": "测试",
        "user": "周知瑞",
        "interface": "Web",
        "jurisdiction": None,
    }
    query = Query(test_inputs1)
