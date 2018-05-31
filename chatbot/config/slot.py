# -*- coding: utf-8 -*-
# @Time    : 5/28/18 16:45
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
from copy import deepcopy


_COMMON_SLOT = {
    "uname": {"value": None, "type": None},
    "uarea": {"value": None, "type": None},
    "utime": {"value": None, "type": None},
}

_POLICY_SLOT = {
    "time": {"value": None, "type": None},
    "area": {"value": None, "type": None},
    "context": {"values": None, "type": None}
}

_DATA_SLOT = {
    "start_time": {"value": None, "type": None},
    "end_time": {"value": None, "type": None},
    "project": {"value": None, "type": None}
}


def get_common_slot():
    return deepcopy(_COMMON_SLOT)


def get_policy_slot():
    return deepcopy(_POLICY_SLOT)


def get_data_slot():
    return deepcopy(_DATA_SLOT)
