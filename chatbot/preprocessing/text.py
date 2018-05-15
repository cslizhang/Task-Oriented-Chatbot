# coding:utf8
# @Time    : 18-5-14 上午9:48
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com


# TODO
def text_precessing(x):
    """特殊符号过滤，标点符号映射

    :param x: String
    :return:  String
    """
    x = text_filter(x)
    x = text_map(x)
    return x


def text_map(x):
    """映射：{？！『』“”：。等}-> {?!\"\"\"\"\".等}

    :param x: String
    :return: String
    """
    pass


def text_filter(x):
    """除中文，英文，数字，常见并支持textmap的标点符号外，全部过滤不要

    :param x: String
    :return: String
    """