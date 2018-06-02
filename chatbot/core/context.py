# coding:utf8
# @Time    : 18-6-2 下午4:48
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import datetime as dt

from chatbot.core.user import User
from chatbot.core.message import Query, Response
from chatbot.config.constant import TIMEOUT


class Context(object):
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
