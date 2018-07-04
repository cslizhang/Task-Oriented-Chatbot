# -*- coding: utf-8 -*-
# @Time    : 5/11/18 11:16
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com

import logging.config
from pathlib import Path

from chatbot.utils.path import ROOT_PATH

DEBUG_LOG = 'debug.log'
INFO_LOG = 'info.log'


def get_debug_log(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    # debug log define
    debug_path = Path(ROOT_PATH, DEBUG_LOG).resolve()
    debug_hdl = logging.FileHandler(debug_path)
    debug_hdl.setLevel(logging.DEBUG)
    debug_hdl.setFormatter(fmt)
    logger.addHandler(debug_hdl)
    # stream log define
    stream_hdl = logging.StreamHandler()
    stream_hdl.setLevel(logging.INFO)
    stream_hdl.setFormatter(fmt)
    logger.addHandler(stream_hdl)
    return logger


def get_info_log(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    # info log define
    info_path = Path(ROOT_PATH, INFO_LOG).resolve()
    info_hdl = logging.FileHandler(info_path)
    info_hdl.setFormatter(fmt)
    info_hdl.setLevel(logging.INFO)
    logger.addHandler(info_hdl)
    # stream log define
    stream_hdl = logging.StreamHandler()
    stream_hdl.setLevel(logging.INFO)
    stream_hdl.setFormatter(fmt)
    logger.addHandler(stream_hdl)
    return logger


def get_logger(logger_name):
    """获取日志输出器
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    # debug log define
    debug_path = Path(ROOT_PATH, DEBUG_LOG).resolve()
    debug_hdl = logging.FileHandler(debug_path)
    debug_hdl.setLevel(logging.DEBUG)
    debug_hdl.setFormatter(fmt)
    logger.addHandler(debug_hdl)
    # info log define
    info_path = Path(ROOT_PATH, INFO_LOG).resolve()
    info_hdl = logging.FileHandler(info_path)
    info_hdl.setFormatter(fmt)
    info_hdl.setLevel(logging.INFO)
    logger.addHandler(info_hdl)
    # stream log define
    stream_hdl = logging.StreamHandler()
    stream_hdl.setLevel(logging.INFO)
    stream_hdl.setFormatter(fmt)
    logger.addHandler(stream_hdl)
    return logger


if __name__ == "__main__":
    l = get_logger("test")
    l.info("info")
    l.warning("warning")
    l.debug("debug")
