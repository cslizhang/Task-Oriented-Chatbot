# coding:utf8
# @Time    : 18-6-2 下午8:00
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from threading import Timer

from chatbot.config.constant import TIMEOUT_CHECK_FREQ
from chatbot.utils.log import get_logger


class BaseBot(object):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

