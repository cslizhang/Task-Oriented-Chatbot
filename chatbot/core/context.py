# coding:utf8
# @Time    : 18-6-2 下午4:48
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import datetime as dt

from chatbot.core.user import User
from chatbot.core.message import Query, Response
from chatbot.config.constant import TIMEOUT


class Context(object):
    """会话上下文，存储一个用户的本轮会话所有信息，本质上是DST(dialog state track)。

    Attributes:
        user <class User>: 用户
        interface <String>: 用户接入的终端，在Query中会指定
        start_time <class datetime.datetime>: 会话开始时间
        last_time <class datetime.datetime>: 该会话用户最后一次请求时间
        history_query <List of class Query>: 该会话历史用户请求，按时间顺序排列
        history_response <List of class Response>: 该会话历史回复，按时间顺序排列
        history_intent <List of class Intent>: 该会话历史意图，按时间顺序排列
        unfinished_action <>: sdf
        context_id <String>: 该会话上下文id
    Methods:
        update (msg): 根据最新的请求或回复更新上下文
        is_timeout: 判断是否超时，超时会话上下文将被删除
    """
    def __init__(self, query):
        """

        :param query: <class Query>
        """
        self.user = User(query.user_name, query.jurisdiction)
        self.interface = query.interface
        self.start_time = query.time
        self.last_time = query.time
        self.history_query = []
        self.history_response = []
        self.current_query = query
        self.context_id = str(hash(str(self.user) + self.interface + query.strtime))

    def update(self, msg):
        """根据最新的Message更新上下文

        :param msg:  <class Query, Response>
        :return: self
        """
        if isinstance(msg, Query):
            self._update_from_query(msg)
        elif isinstance(msg, Response):
            self._update_from_response(msg)
        else:
            raise ValueError
        return self

    def _update_from_query(self, query):
        self.history_query.append(query)
        self.current_query = query

    def _update_from_response(self, resp):
        self.history_response.append(resp)

    @property
    def is_timeout(self):
        """是否超时

        如果超时，上下文将被从DM中删除
        """
        if (dt.datetime.now() - self.last_time).seconds > TIMEOUT:
            return True
        else:
            return False
