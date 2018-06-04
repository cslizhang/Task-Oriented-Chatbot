# coding:utf8
# @Time    : 18-6-2 下午5:28
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com


class User(object):
    def __init__(self, name, jurisdiction=None):
        """

        :param name: <String>
        # TODO
        :param jurisdiction: <Not defined now>
        """
        self.name = name
        self.jurisdiction = jurisdiction

    def __str__(self):
        return "Class User <name: {}>, <jurisdiction>: {}>".format(
            self.name,
            str(self.jurisdiction)
        )

    def __repr__(self):
        return self.__str__()
