# coding:utf8
# @Time    : 18-6-7 上午10:51
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com

import calendar
import re
import datetime

from chatbot.core.estimator import Estimator
from chatbot.core.entity import TimeInterval


class NerRuleV1:
    def __init__(self):
        super().__init__()

    def extract(self, context):
        """
        :param context: context["query"]
        :return: <dict of list>
        {"TimeInterval": ["", ""]}

        """
        rst = {}
        ext_time = self._extract_time(context)
        if ext_time is not None:
            rst["TimeInterval"] = ext_time
        return rst

    def _extract_time(self, context):
        """
        提取时间实体
        :param context: context["query"]
        :return: 假如没有实体，返回None，
        """
        pattern1 = [r'(\d{4}-\d{1,2}-\d{1,2})']
        pattern2 = [r'(\d{4}/\d{1,2}/\d{1,2})']
        pattern3 = [r'(\d{2,4}\.\d{1,2}\.\d{1,2})|(\d{2,4}\.\d{1,2})|(\d{1,2}\.\d{1,2})']
        pattern4 = [r'(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2})']
        pattern5 = [
            r'(([去年|明年|后年|前年|今年].*[月].*[日|号])|([去年|明年|后年|前年|今年].*[月])|([今|明|昨|去|前|后][天|年])|(\d{2,4}[年].*[季度])|(\d{4}[年][第].[季度].)|([第].[季度].)|(.[下个|上个|这个][月].*[日|号])|([本][月].*[日|号])|([\d|一|二|三|四][季度].)|[上|下|本][月|[年|月])|(.[下个|上个|这个][月])|(\d{2,4}年\d{1,2}月\d{1,2}[日|号])|(\d{2,4}年\d{1,2}月)|(\d{1,2}月\d{1,2}[日|号])|([一|二|三|四|五|六|七|八|九|十|十一|十二]{1,2}月\d{1,2}[日|号])|(\d{1,2}[月])|([一|二|三|四|五|六|七|八|九|十|十一|十二]{1,2}[月])|(\d{1,2}[号|日])|([一|二|三|四|五|六|七|八|九|十][号|日])|(\d{2,4}[年])']
        pattern6 = [
            r'([上|下|这|本][周][[一|二|三|四|五|六|天|\d])|([上|下|这|本][星期|礼拜].. *?)|([上一|下一].[周|天|年])|([上|下|这|本][周])|([周].)']
        pattern7 = [r'(.[下个|上个|这个][星期|礼拜].[\d|一|二三|四|五|六|七|天])|(.[下个|上个|这个][星期|礼拜].)|([本|下|上][季度].)|(.[下个|上个|这个][季度].)']
        pattern = pattern1 + pattern2 + pattern3 + pattern4 + pattern5 + pattern6 + pattern7

        '''正则匹配'''
        result_origin = []
        for i in range(len(pattern)):
            res = re.findall(pattern[i], context['query'], re.S)
            if len(res) != 0:
                result_origin.append(res)
            else:
                continue
        # print (result_origin)
        '''处理输出结果的格式'''
        extract_result = []
        for i in range(len(result_origin)):
            if len(result_origin) > 1:
                if type(result_origin[i][0]) == str:
                    for k in range(len(result_origin[i])):
                        extract_result.append(result_origin[i][k])
                if type(result_origin[i][0]) == tuple:
                    for j in range(len(result_origin[i])):
                        a = list(result_origin[i][j])
                        while '' in a:
                            a.remove('')
                        extract_result.append(a[0])
            else:
                if type(result_origin[0][0]) == str:
                    extract_result.append(result_origin[0][0])
                if type(result_origin[0][0]) == tuple:
                    for q in range(len(result_origin[0])):
                        a = list(result_origin[0][q])
                        while '' in a:
                            a.remove('')
                        extract_result.append(a[0])
        if len(extract_result) == 0:
            return None
        else:
            return extract_result

    def transform(self):
        pass

    def _infer_time_entity(self, context):
        """

        :param context:
        :return: <list of time entity>
        """
        pass

    def _infer_location_entity(self, context):
        pass


if __name__ == "__main__":
    context = {'query':'上周五用电量'}
    contexts = [
        {'query': '上周五用电量'},
        {'query': '2018年3月，2018年到2019年电量'},
        {'query': '2017用电量'},
        {'query': '上个月'},
        {'query': '上周五用电量'},
        {'query': '上周五用电量'},
        {'query': '上周五用电量'},
    ]
    ner = NerRuleV1()
    for c in contexts:
        print(ner.extract(c))
    # d = a.transform(context)
    # print(d)
