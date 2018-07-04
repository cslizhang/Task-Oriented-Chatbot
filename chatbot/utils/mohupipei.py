# coding:utf8
# @Time    : 18-7-3 下午4:20
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com


def fuzzy_substring(needle, haystack):
    """Calculates the fuzzy match of needle inhaystack,
    using a modified version of the Levenshtein distance
    algorithm.
    The function is modified from the levenshtein function
    in the bktree module by Adam Hupp"""

    m, n = len(needle), len(haystack)

    # base cases
    if m == 1:
        return not needle in haystack
    if not n:
        return m

    row1 = [0] * (n+1)
    for i in range(0,m):
        row2 = [i+1]
        for j in range(0,n):
            cost = ( needle[i] != haystack[j] )

            row2.append(min(row1[j+1]+1, # deletion
                              row2[j]+1, #insertion
                              row1[j]+cost) #substitution
                          )
        row1 = row2
    return min(row1)


if __name__ == "__main__":
    key = "来福士"
    s1 = "哈哈，来f士"
    s2 = "来"
    fuzzy_substring(key, s1)
    fuzzy_substring(key, s2)