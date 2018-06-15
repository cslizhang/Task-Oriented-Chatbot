# coding:utf8
# @Time    : 18-6-7 上午10:51
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com


import re


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

    @staticmethod
    def _extract_time(context):
        """
        提取时间实体
        :param context: context["query"]
        :return: 假如没有实体，返回None，
        """

        pattern = [
            r'(\d{2,4}-\d{1,2}-\d{1,2})|(\d{8})|(\d{6})|(\d{2,4}/\d{1,2}/\d{1,2})|(\d{2,4}\.\d{1,2}\.\d{1,2})|'
            r'(\d{2,4}\.\d{1,2})|(\d{1,2}\.\d{1,2})|(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2})|(\d{2,4}年\d{1,2}月)|'
            r'(\d{2,4}年\d{1,2}月\d{1,2}[日|号])|(.[去年|明年|后年|前年|今年][\d|一|二|三|四|五|六|七|八|九|十][月].\d[日|号])|'
            r'(.[去年|明年|后年|前年|今年]\d{1,2}[月])|([去年|明年|后年|前年|今年](.*)季度)|([今|明|昨|去|前|后][年][今|明|昨|前|后][天|日])|'
            r'([今|明|昨|去|前|后][天|年|日])|(\d{2,4}[年].*[季度]$)|(\d{4}[年][第].[季度].)|([第].[季度].)|([\d|一|二|三|四][季度].)|'
            r'([本|上|下][月].*[日|号])|(^[下个月|上个月|这个月].*\d{2}[日|号])|([上|下|本][月|年])|([下个|上个|这个].[月])|'
            r'([过去|未来].*[月|天])|(\d{2,4}年\d{1,2}月)|(\d{1,2}月\d{1,2}[日|号])|'
            r'([一|二|三|四|五|六|七|八|九|十|十一|十二]{1,2}月\d{1,2}[日|号])|(\d{1,2}[月])|([一|二|三|四|五|六|七|八|九|十|十一|十二]{1,2}[月])|'
            r'(\d{1,2}[号|日])|([一|二|三|四|五|六|七|八|九|十][号|日])|(\d{2,4}[年])|([上|下|这|本][周][[一|二|三|四|五|六|天|\d])|'
            r'([上|下|这|本][星期|礼拜].. *?)|(^[上一|下一].[周|天|年])|([上|下|这|本][周])|([周][\d|一|二|三|四|五])|'
            r'(.[下个|上个|这个][星期|礼拜].[\d|一|二三|四|五|六|七|天])|(.[下个|上个|这个][星期|礼拜].)|([本|下|上][季度].)|(.[下个|上个|这个][季度].)']

        result_origin = []
        for i in range(len(pattern)):
            res = re.findall(pattern[i], context['query'], re.S)
            if len(res) != 0:
                result_origin.append(res)
            else:
                continue
        extract_result = []
        for i in range(len(result_origin)):
            if len(result_origin) > 1:
                if isinstance(result_origin[i][0], str):
                    for k in range(len(result_origin[i])):
                        extract_result.append(result_origin[i][k])
                if isinstance(result_origin[i][0], tuple):
                    for j in range(len(result_origin[i])):
                        res_a = list(result_origin[i][j])
                        while '' in a:
                            res_a.remove('')
                        extract_result.append(res_a[0])
            else:
                if isinstance(result_origin[0][0], str):
                    extract_result.append(result_origin[0][0])
                if isinstance(result_origin[0][0], tuple):
                    for q in range(len(result_origin[0])):
                        res_deal = list(result_origin[0][q])
                        while '' in res_deal:
                            res_deal.remove('')
                        extract_result.append(res_deal[0])
        if len(extract_result) == 0:
            return None
        else:
            return extract_result

    def transform(self):
        pass

    @staticmethod
    def _infer_time_entity(self, context):
        """

        :param context:
        :return: <list of time entity>
        """
        pass

    def _infer_location_entity(self, context):
        pass


if __name__ == "__main__":
    # with open("/home/zhouzr/ner_time_extract.txt", 'r') as f:
    #     d = f.read().split("\t")
    # text = []
    # text += ["上周二用电"]
    # for idx, i in enumerate(d):
    #     if (idx - 1) % 3 == 0:
    #         text.append(i)
    # context = {'query':'帮我查下前天就是25号的电量'}
    # contexts = [{"query": i} for i in text]
    # ner = NerRuleV1()
    # for c in contexts:
    #     print(c["query"], ner.extract(c), "\n")
    for line in open("test.txt", 'r'):
        contexts = dict()
        contexts['query'] = line.split(' ')[0]
        print(contexts)
        a = NerRuleV1()
        b = a.extract(contexts)
        print(b)
        if str(b['TimeInterval']).replace(', ', ',') == line.split(' ')[1].strip('\n'):
            print('True', '\n')
        else:
            print('False', '\n')
