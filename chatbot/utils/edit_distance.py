# coding:utf8
# @Time    : 18-7-4 上午9:19
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import numpy as np


def substring_edit_distance(key_word, query):
    """计算子串最小编辑距离函数，包含删除/添加/替换三种编辑操作

    Args:
        key_word:
        query:

    Returns:
        int

    Example:

    >>> isinstance(substring_edit_distance("来福士", "sdf来福斯蒂芬"), int)
    True

    >>> substring_edit_distance("abc", "c aba c")
    1

    >>> substring_edit_distance("来福士", "sdf福士斯蒂芬")
    1

    """
    len_query = len(query)
    len_keyword = len(key_word)
    matrix = np.zeros([len_keyword+1, len_query+1])
    matrix[:, 0] = range(len_keyword+1)
    for row in range(len_keyword):
        for col in range(len_query):
            replace = 0 if query[col] == key_word[row] else 1
            compare = min(
                matrix[row][col+1] + 1,
                matrix[row+1][col] + 1,
                matrix[row][col] + replace,
            )
            matrix[row+1][col+1] = compare
    return int(min(matrix[-1, :]))


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
