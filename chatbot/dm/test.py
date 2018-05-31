# -*- coding: utf-8 -*-
# @Time    : 5/28/18 15:17
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com

from threading import Timer
import time


class DM:
    def __init__(self):
        self.last_query_time = time.time()
        self.label = True
        self.__count()

    def __count(self):
        if time.time() - self.last_query_time <5:
            pass
        else:
            self.label=False

        timer=Timer(10, self.__count)
        timer.start()

    def terminal_chatbot(self):
        while self.label:
            s = input()
            self.last_query_time = time.time()
            print(s)

from chatbot.config.slot import *
