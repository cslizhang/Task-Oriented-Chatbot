# coding:utf8
# @Time    : 18-6-2 下午3:12
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import datetime as dt


class BaseMessage(object):
    def __init__(self, text, user_name, interface):
        """消息初始化

        :param text: <String>
        :param user_name: <String>
        :param interface: <String>
        """
        self.user_name = user_name
        self.text = text
        self.interface = interface
        self.time = dt.datetime.now()
        self.strtime = self.time.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def type(self):
        raise NotImplementedError

    def __str__(self):
        return "A {} Message <user: {}> <time: {}> <interface: {}> <text: {}>".format(
            self.type,
            self.user_name,
            self.time,
            self.interface,
            self.text
        )

    def __repr__(self):
        return self.__str__()

    # TODO: 消息检查
    def check_state(self):
        pass


class Response(BaseMessage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def type(self):
        return "Response"


class Query(BaseMessage):
    def __init__(self, jurisdiction=None, **kwargs):
        super().__init__(**kwargs)
        self.jurisdiction = jurisdiction

    @property
    def type(self):
        return "Query"



