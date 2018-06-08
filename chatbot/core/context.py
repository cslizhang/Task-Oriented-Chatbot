# coding:utf8
# @Time    : 18-6-2 下午4:48
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import datetime as dt

from chatbot.config.constant import TIMEOUT


class Context(dict):
    def __init__(self, user, app, skill2slot, context_id, right=None, timeout=TIMEOUT):
        self._timeout = timeout
        now = dt.datetime.now()
        super().__init__(
            user=user,
            app=app,
            right=right if right else [],
            history_query=[],
            history_resp=[],
            history_intent=[],
            query=None,
            query_cut=None,
            intent=None,
            entities=None,
            slots=skill2slot,
            last_query_time=now,
            context_id=context_id
        )

    @property
    def is_timeout(self):
        """是否超时

        如果超时，上下文将被从DM中删除
        """
        if (dt.datetime.now() - self["last_query_time"]).seconds > self._timeout:
            return True
        else:
            return False


if __name__ == "__main__":
    c = Context(user="zhouzr", app="web2.0", skill2slot=dict(), context_id="sad")
