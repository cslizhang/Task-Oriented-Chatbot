# coding:utf8
# @Time    : 18-5-31 下午2:22
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import time
from threading import Timer

from chatbot.utils.log import get_logger


class Bot(object):
    def __init__(self, timeout=60, timeout_check_freq=1):
        """Chatbot服务初始化

        Args:
            timeout (Float, Int), 单位秒，超过timeout的会话上下文将被清空
            timeout_check_freq (Float, Int), 单位次每秒，检查timeout的频率
        """
        self.timeout = timeout
        self.timeout_check_freq = timeout_check_freq
        self.sessions = dict()
        self._delete_timeout_session()
        self.logger = get_logger(self.__class__.__name__)

    def chat(self):
        """传入用户对话文本，返回chatbot回复

        1.如果没有对应的session，则创建
        2.更新query对应的session最新的对话时间
        3.开始会话处理

        Args:
            query (Dict):
            + user_name (String), 用户名
            + text (String), 用户输入文本
            + access_type (String), 用户接入类型
        Returns:
            response (Dict)
        """
        # TODO
        while True:
            terminal_input = input()
            text = terminal_input.split(" ")[0]
            user_name = terminal_input.split(" ")[1]
            query = {"user_name": user_name,
                     "text": text,
                     "access_type": "terminal"}
            sid = self._get_session_id(query)
            if self._exist_session(sid):
                self._query_update_session(query, sid)
                self.logger.info("update session sid: {}".format(sid))
            else:
                session = self._init_session(query)
                self.sessions[sid] = session
                self.logger.info("add session sid: {}".format(sid))
            self.logger.info("{}".format(str(self.sessions)))

    def _delete_timeout_session(self):
        current_time = time.time()
        sids = list(self.sessions.keys())
        for sid in sids:
            last_query_time = self.sessions[sid].get("last_query_time", current_time)
            if current_time - last_query_time > self.timeout:
                self._remove_session(sid)
                self.logger.info("timeout remove sid: {}".format(sid))
        timer = Timer(self.timeout_check_freq, self._delete_timeout_session)
        timer.start()

    def _remove_session(self, sid):
        try:
            del self.sessions[sid]
        except KeyError as e:
            # TODO
            pass

    @staticmethod
    def _get_session_id(query):
        sid = "_".join([
            query.get("user_name", "default"),
            query.get("access_type", "default")
        ])
        return sid

    def _exist_session(self, sid):
        if sid in self.sessions.keys():
            return True
        else:
            return False

    def _query_update_session(self, query, sid):
        self.sessions[sid]["history_text"].append(query["text"])
        self.sessions[sid]["current_text"] = query["text"]
        self.sessions[sid]["last_query_time"] = time.time()

    @staticmethod
    def _init_session(query):
        session = {
            "current_text": query["text"],
            "history_text": [query["text"]],
            "user_name": query["user_name"],
            "last_query_time": time.time(),
        }
        return session


if __name__ == "__main__":
    query_1 = {"user_name": "test1", "text": "哈哈哈", "access_type": "terminal"}
    query_2 = {"user_name": "test2", "text": "哈哈哈", "access_type": "terminal"}
    bot = Bot(timeout=10, timeout_check_freq=1)
    bot.chat()
