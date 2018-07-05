# coding:utf8
# @Time    : 18-6-7 上午10:51
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import calendar
import re
import datetime
from chatbot.core.entity import TimeInterval, Location


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
        ext_location = self._infer_location_entity(context)
        if ext_time is not None:
            rst[TimeInterval.name()] = ext_time
        if ext_location is not None:
            rst[Location.name()] = ext_location
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
            r'(\d{2,4}年\d{1,2}月\d{1,2}[日|号])|'
            r'(.[去年|明年|后年|前年|今年][\d|一|二|三|四|五|六|七|八|九|十][月].\d[日|号])|(.[去年|明年|后年|前年|今年]\d{1,2}[月])|'
            r'([去年|明年|后年|前年|今年](.*)季度)|([今|明|昨|去|前|后][年][今|明|昨|前|后][天|日])|([今|明|昨|去|前|后][天|年|日])|'
            r'(\d{2,4}[年][\d|一|二|三|四][季度].)|(\d{4}[年][第].[季度].)|([第].[季度].)|'
            r'([\d|一|二|三|四][季度].)|([本|上|下][月].*[日|号])|'
            r'(^[下个月|上个月|这个月].*\d{2}[日|号])|([上|下|本][月|年])|([下个|上个|这个].[月])|([过去|未来|最近].*?[年|月|天|日])|'
            r'(\d{2,4}年\d{1,2}月)|(\d{1,2}月\d{1,2}[日|号])|([一|二|三|四|五|六|七|八|九|十|十一|十二]{1,2}月\d{1,2}[日|号])|'
            r'(\d{1,2}[月])|([一|二|三|四|五|六|七|八|九|十|十一|十二]{1,2}[月])|(\d{1,2}[号|日])|([一|二|三|四|五|六|七|八|九|十][号|日])|'
            r'(\d{2,4}[年])|([上|下|这|本][周][[一|二|三|四|五|六|天|\d])|([上|下|这|本][星期|礼拜].. *?)|(^[上一|下一].[周|天|年])|'
            r'([上|下|这|本][周])|([周][\d|一|二|三|四|五])|(.[下个|上个|这个][星期|礼拜].[\d|一|二三|四|五|六|七|天])|'
            r'(.[下个|上个|这个][星期|礼拜].)|([本|下|上][季度].)|(.[下个|上个|这个][季度].)|(最新)']

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

    def transform(self, context):
        """
        :param context: context['query']
        :return: dict of list
        [{'end':'2018-06-13','start':'2018-06-13'}]
        """
        rst = {}
        time_output_result = []
        loc_output_result = []
        trans_time = self._infer_time_entity(context)
        trans_location = self._infer_location_entity(context)
        if trans_time is not None:
            for i in range(len(trans_time)):
                transform = TimeInterval()
                if '~' in trans_time[i]:
                    transform['end'] = trans_time[i].split('~')[1]
                    transform['start'] = trans_time[i].split('~')[0]
                else:
                    transform['end'] = trans_time[i]
                    transform['start'] = trans_time[i]
                time_output_result.append(transform)
                rst[TimeInterval.name()] = time_output_result[0]
        if trans_location is not None:
            transform_2 = Location()
            if 'province' in trans_location[0]:
                transform_2['province'] = trans_location[0]['province']
            if 'city' in trans_location[0]:
                transform_2['city'] = trans_location[0]['city']
            loc_output_result.append(transform_2)
            rst[Location.name()] = loc_output_result[0]
        if trans_time is not None or trans_location is not None:
            return rst
        else:
            return None

    def _infer_time_entity(self, context):
        """
        :param context:
        :return: <list of time entity>
        """
        res = self._extract_time(context)
        res1 = self._infer_time_standard(context)
        res2 = self._infer_time_anglicize(context)
        res3 = self._infer_time_mixed(context)
        res4 = self._infer_time_contact(context)
        if res is not None:
            if len(res4) == 0:
                res = res1 + res2 + res3
            elif len(res4) == 1 and len(res2) == 2:
                res = res4
            elif len(res4) == 1 and len(res3) == 2:
                res = res4
            else:
                res = res4 + res2 + res3
            return res
        else:
            return None

    @staticmethod
    def _has_numbers(string):
        return any(char.isdigit() for char in string)

    @staticmethod
    def _num_transform(i):
        """中文大写数字转换阿拉伯数字"""
        if ('一' in i or '二' in i or '两' in i or '三' in i or '四' in i or '五' in i or '六' in i or
                '七' in i or '八' in i or '九' in i or '十' in i or '天' in i):
            try:
                i = i.replace('十一', '11')
                i = i.replace('十二', '12')
                i = i.replace('一', '1')
                i = i.replace('二', '2')
                i = i.replace('两', '2')
                i = i.replace('三', '3')
                i = i.replace('四', '4')
                i = i.replace('五', '5')
                i = i.replace('六', '6')
                i = i.replace('七', '7')
                if '星期' in i or '礼拜' in i or '周' in i:
                    i = i.replace('天', '7')
                i = i.replace('八', '8')
                i = i.replace('九', '9')
                i = i.replace('十', '10')
            except Exception as e:
                print(e)
        return i

    def _infer_time_standard(self, context):
        """
        标准型时间映射
        :param context:
        :return:
        """
        result = self._extract_time(context)
        if result is not None:
            convert_result1 = []
            # user_param = TEST()
            for i in result:
                i = self._num_transform(i)
                if self._has_numbers(i) is True and '周' not in i and '星期' not in i and '礼拜' not in i and '季' not in i \
                        and '去年' not in i and '今年' not in i and '前年' not in i and '后年' not in i and '明年' not in i \
                        and '上' not in i and '下' not in i and '本' not in i and '这' not in i and '过去' not in i \
                        and '未来' not in i and '最近' not in i:
                    days = ''
                    years = ''
                    months = ''
                    time_convert = ''
                    '''时间格式的映射'''
                    if '.' in i or '-' in i or '/' in i:
                        if '.' in i:
                            if len(i.split('.')) == 3:
                                if len(i.split('.')[0]) == 4:
                                    years = i.split('.')[0]
                                if len(i.split('.')[0]) == 2:
                                    years = str(2000 + int(i.split('.')[0]))
                                if int(
                                        i.split('.')[1]) < 10 and len(
                                        i.split('.')[1]) == 1:
                                    months = '0' + (i.split('.')[1])
                                if len(i.split('.')[1]) == 2:
                                    months = i.split('.')[1]
                                if int(i.split('.')[2]) < 10 and len(i.split('.')[2]) == 1:
                                    days = '0' + (i.split('.')[2])
                                if len(i.split('.')[2]) == 2:
                                    days = i.split('.')[2]
                            elif len(i.split('.')) == 2:
                                if len(i.split('.')[0]) == 4:
                                    years = i.split('.')[0]
                                    months = i.split('.')[1]
                                    if int(months) < 10 and len(months) == 1:
                                        months = '0' + months
                                    else:
                                        months = months
                                    month_range = calendar.monthrange(
                                        int(years), int(months))
                                    days = ['01', str(month_range[1])]
                                if len(i.split('.')[0]) == 2:
                                    years = str(datetime.datetime.now().year)
                                    months = i.split('.')[0]
                                    if int(months) < 10 and len(months) == 1:
                                        months = '0' + months
                                    else:
                                        months = months
                                    if int(i.split('.')[1]) < 10 and len(i.split('.')[1]) == 1:
                                        days = '0' + (i.split('.')[1])
                                    else:
                                        days = i.split('.')[1]
                        if '/' in i:
                            if len(i.split('/')[0]) == 4:
                                years = i.split('-')[0]
                            if len(i.split('/')[1]) == 2:
                                years = str(2000 + int(i.split('/')[0]))
                            if int(i.split('/')
                                   [1]) < 10 and len(i.split('/')[1]) == 1:
                                months = '0' + (i.split('/')[1])
                            if len(i.split('/')[1]) == 2:
                                months = i.split('/')[1]
                            if int(i.split('/')
                                   [2]) < 10 and len(i.split('/')[2]) == 1:
                                days = '0' + (i.split('/')[2])
                            if len(i.split('/')[2]) == 2:
                                days = i.split('/')[2]
                        if '-' in i:
                            if len(i.split('-')[0]) == 4:
                                years = i.split('-')[0]
                            if len(i.split('-')[0]) == 2:
                                years = str(2000 + int(i.split('-')[0]))
                            if int(i.split('-')
                                   [1]) < 10 and len(i.split('-')[1]) == 1:
                                months = '0' + (i.split('-')[1])
                            if len(i.split('-')[1]) == 2:
                                months = i.split('-')[1]
                            if int(i.split('-')
                                   [2]) < 10 and len(i.split('-')[2]) == 1:
                                days = '0' + (i.split('-')[2])
                            if len(i.split('-')[2]) == 2:
                                days = i.split('-')[2]
                        if isinstance(days, str):
                            time_convert = years + '-' + months + '-' + days
                        else:
                            res = [years + '-' + months + '-' + day for day in days]
                            time_convert = res[0] + '~' + res[1]
                    if i.isdigit() is True and len(i) == 8:
                        years = i[0:4]
                        months = i[4:6]
                        days = i[6:8]
                        time_convert = years + '-' + months + '-' + days
                    if i.isdigit() is True and len(i) == 6:
                        years = i[0:4]
                        months = i[4:6]
                        month_range = calendar.monthrange(
                            int(years), int(months))
                        days = ['01', str(month_range[1])]
                        res = [years + '-' + months + '-' + day for day in days]
                        time_convert = res[0] + '~' + res[1]

                    '''年月日格式的映射'''
                    if '年' in i or '月' in i or '日' in i or '号' in i:
                        if '年' in i and '月' not in i and '日' not in i and '号' not in i:
                            time_convert = str(
                                i.split('年')[0]) + '-01-01' + '~' + str(i.split('年')[0]) + '-12-31'
                        else:
                            if '年' in i:
                                find_index_year = i.index('年')
                                find_year = i[0:int(find_index_year)]
                                if len(find_year) == 4:
                                    years = find_year
                                if len(find_year) < 4:
                                    years = str(2000 + int(find_year))
                            else:
                                years = str(datetime.datetime.now().year)
                            if '月' in i:
                                try:
                                    find_index_year = i.index('年')
                                except ValueError:
                                    find_index_year = -1
                                find_index_month = i.index('月')
                                find_month = i[int(
                                    find_index_year) + 1:int(find_index_month)]
                                months = find_month
                            else:
                                months = str(datetime.datetime.now().month)
                            if int(months) < 10 and len(months) == 1:
                                months = '0' + months
                            else:
                                months = months
                            if '日' in i or '号' in i:
                                try:
                                    find_index_month = i.index('月')
                                except ValueError:
                                    find_index_month = -1
                                try:
                                    find_index_days = i.index('日')
                                except ValueError:
                                    find_index_days = i.index('号')
                                find_days = i[int(
                                    find_index_month) + 1:int(find_index_days)]
                                days = find_days
                                if int(days) < 10 and len(days) == 1:
                                    days = '0' + days
                                else:
                                    days = days
                            else:
                                month_range = calendar.monthrange(int(years), int(months))
                                days = ['01', str(month_range[1])]
                            if isinstance(days, list):
                                res = [years + '-' + months + '-' + day for day in days]
                                time_convert = res[0] + '~' + res[1]
                            else:
                                time_convert = years + '-' + months + '-' + days
                    convert_result1.append(time_convert)
            return convert_result1
        else:
            return None

    def _infer_time_mixed(self, context):
        """
        混合型时间映射
        :param context:
        :return:
        """
        result = self._extract_time(context)
        if result is not None:
            convert_result = []
            time_convert = ''
            years = ''
            months = ''
            for i in result:
                i = self._num_transform(i)
                if (self._has_numbers(i) is True and
                        ('最近' in i or '本年' in i or '去年' in i or '今年' in i or '前年' in i
                         or '后年' in i or '明年' in i or '上' in i or '下' in i or '本' in i
                         or '这' in i or '过去' in i or '未来' in i)) or (self._has_numbers(i) is True and '季' in i):
                    if '年' in i and '月' in i and '日' not in i and '号' not in i:
                        months = re.search(r'(\d{1,2})月', i, re.S).group(1)
                        if int(months) < 10 and len(months) == 1:
                            months = '0' + months
                        else:
                            months = months
                        years = str(datetime.datetime.now().year)
                        month_range = calendar.monthrange(
                            int(years), int(months))
                        days = ['01', str(month_range[1])]
                        if '今年' in i:
                            years = years
                        if '明年' in i:
                            years = str(datetime.datetime.now().year + 1)
                        if '前年' in i:
                            years = str(datetime.datetime.now().year - 2)
                        if '后年' in i:
                            years = str(datetime.datetime.now().year + 2)
                        if '去年' in i:
                            years = str(datetime.datetime.now().year - 1)
                        res = [
                            years +
                            '-' +
                            months +
                            '-' +
                            day for day in days]
                        time_convert = res[0] + '~' + res[1]
                    if '年' in i and '月' in i and ('日' in i or '号' in i):
                        months = re.search(r'(\d{1,2})月', i, re.S).group(1)
                        days = re.search(r'(\d{1,2})[月|日]', i, re.S).group(1)
                        if int(months) < 10 and len(months) == 1:
                            months = '0' + months
                        else:
                            months = months
                        if int(days) < 10 and len(days) == 1:
                            days = '0' + days
                        else:
                            days = days
                        years = str(datetime.datetime.now().year)
                        if i == '今年':
                            years = years
                        if i == '明年':
                            years = str(datetime.datetime.now().year + 1)
                        if i == '前年':
                            str(datetime.datetime.now().year - 2)
                        if i == '后年':
                            str(datetime.datetime.now().year + 2)
                        if i == '去年':
                            str(datetime.datetime.now().year - 1)
                        time_convert = years + '-' + months + '-' + days
                    if '年' not in i and '月' in i and ('日' in i or '号' in i):
                        months = datetime.datetime.now().month
                        days = re.search(r'(\d{1,2})[日|号]', i, re.S).group(1)
                        if int(days) < 10 and len(days) == 1:
                            days = '0' + days
                        else:
                            days = days
                        years = str(datetime.datetime.now().year)
                        if '上' in i:
                            months = str(months - 1)
                            if int(months) < 10 and len(months) == 1:
                                months = '0' + months
                            else:
                                months = months
                        if '本' in i or '这' in i:
                            if int(months) < 10 and len(months) == 1:
                                months = '0' + months
                            else:
                                months = months
                        if '下' in i:
                            months = str(months + 1)
                            if int(months) < 10 and len(months) == 1:
                                months = '0' + months
                            else:
                                months = months
                        time_convert = years + '-' + months + '-' + days
                    if '年' in i and '季' in i:
                        if self._has_numbers(i.split('年')[0]) is True:
                            years = re.search(
                                r'(\d{2,4})[年]', i, re.S).group(1)
                            if len(years) == 4:
                                years = years
                            else:
                                years = str(2000 + int(years))
                        else:
                            if '今年' in i:
                                years = str(datetime.datetime.now().year)
                            if '明年' in i:
                                years = str(datetime.datetime.now().year + 1)
                            if '去年' in i:
                                years = str(datetime.datetime.now().year - 1)
                            if '后年' in i:
                                years = str(datetime.datetime.now().year + 2)
                            if '前年' in i:
                                years = str(datetime.datetime.now().year - 2)

                        num = int(re.search(r'(\d)[季]', i, re.S).group(1))
                        if num == 1:
                            months = ['01', '03']
                        if num == 2:
                            months = ['04', '06']
                        if num == 3:
                            months = ['07', '09']
                        if num == 4:
                            months = ['10', '12']
                        month_range = calendar.monthrange(
                            int(years), int(months[1]))
                        time_convert = years + '-' + \
                            months[0] + '-01' + '~' + years + '-' + months[1] + '-' + str(month_range[1])
                    if '年' not in i and '季' in i:
                        num = int(re.search(r'(\d)[季]', i, re.S).group(1))
                        if num == 1:
                            months = ['01', '02', '03']
                        if num == 2:
                            months = ['04', '05', '06']
                        if num == 3:
                            months = ['07', '08', '09']
                        if num == 4:
                            months = ['10', '11', '12']
                        years = str(datetime.datetime.now().year)
                        res = [years + '-' + x for x in months]
                        month_range = calendar.monthrange(
                            int(years), int(months[2]))
                        days = [str(month_range[1])]
                        time_convert = res[0] + '-01' + \
                            '~' + res[2] + '-' + days[0]
                    if '周' in i or '星期' in i or '礼拜' in i:
                        if '上' in i:
                            week_day = int(
                                datetime.datetime.now().isoweekday())
                            find_week = int(i[-1])
                            d_day = - week_day - (7 - find_week)
                            time_convert = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=d_day)).strftime('%Y-%m-%d')
                        if '本' in i or '这' in i:
                            week_day = int(
                                datetime.datetime.now().isoweekday())
                            find_week = int(i[-1])
                            d_day = find_week - week_day
                            time_convert = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=d_day)).strftime('%Y-%m-%d')
                        if '下' in i:
                            week_day = int(
                                datetime.datetime.now().isoweekday())
                            find_week = int(i[-1])
                            d_day = (7 - week_day) + find_week
                            time_convert = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=d_day)).strftime('%Y-%m-%d')
                    if '过去' in i or '最近' in i:
                        years = str(datetime.datetime.now().year)
                        months = datetime.datetime.now().month
                        if months < 10 and len(str(months)) == 1:
                            months = '0' + str(months)
                        else:
                            months = months
                        if '月' in i:
                            num = re.search(
                                r'(\d{1,2})月', i.replace(
                                    '个', ''), re.S).group(1)
                            month_start = (datetime.datetime.now() + datetime.timedelta(days=-int(num)*30)).month
                            month_end = (datetime.datetime.now() + datetime.timedelta(days=-30)).month
                            year_start = (datetime.datetime.now() + datetime.timedelta(days=-int(num)*30)).year
                            year_end = (datetime.datetime.now() + datetime.timedelta(days=-30)).year
                            if month_start < 10 and len(str(month_start)) == 1:
                                month_start = '0' + str(month_start)
                            else:
                                month_start = month_start
                            if month_end < 10 and len(str(month_end)) == 1:
                                month_end = '0' + str(month_end)
                            else:
                                month_end = month_end
                            month_range = calendar.monthrange(
                                int(year_end), int(month_end))
                            time_convert = str(year_start) + '-' + str(month_start) + '-01' + '~' + \
                                str(year_end) + '-' + str(month_end) + '-' + str(month_range[1])
                        if '日' in i or '天' in i:
                            num = re.search(
                                r'(\d{1,2})[日|天]', i, re.S).group(1)
                            start_day = (datetime.datetime.now() + datetime.timedelta(days=-int(num))).day
                            end_day = (datetime.datetime.now() + datetime.timedelta(days=-1)).day
                            start_month = (datetime.datetime.now() + datetime.timedelta(days=-int(num))).month
                            end_month = (datetime.datetime.now() + datetime.timedelta(days=-1)).month
                            time_convert = str(years) + '-' + str(start_month) + '-' + str(
                                start_day) + '~' + str(years) + '-' + str(end_month) + '-' + str(end_day)
                    if '最近' in i and '年' in i:
                        num = re.search(r'(\d{1,2})年', i, re.S).group(1)
                        start_years = str(
                            datetime.datetime.now().year - int(num))
                        end_years = str(datetime.datetime.now().year)
                        month_range = calendar.monthrange(int(years), 12)
                        time_convert = start_years + '-' + '01-01' + '~' + \
                            end_years + '-' + '12' + '-' + str(month_range[1])
                    if '未来' in i:
                        years = str(datetime.datetime.now().year)
                        months = datetime.datetime.now().month
                        if months < 10 and len(str(months)) == 1:
                            months = '0' + str(months)
                        else:
                            months = months
                        if '月' in i:
                            num = re.search(
                                r'(\d{1,2})月', i.replace(
                                    '个', ''), re.S).group(1)
                            month_start = (datetime.datetime.now() + datetime.timedelta(days=30)).month
                            month_end = (datetime.datetime.now() + datetime.timedelta(days=int(num)*30)).month
                            year_start = (datetime.datetime.now() + datetime.timedelta(days=30)).year
                            year_end = (datetime.datetime.now() + datetime.timedelta(days=int(num) * 30)).year
                            if month_start < 10 and len(str(month_start)) == 1:
                                month_start = '0' + str(month_start)
                            else:
                                month_start = month_start
                            if month_end < 10 and len(str(month_end)) == 1:
                                month_end = '0' + str(month_end)
                            else:
                                month_end = month_end
                            month_range = calendar.monthrange(
                                int(year_end), int(month_end))
                            time_convert = str(year_start) + '-' + str(month_start) + '-01' + '~' + str(
                                year_end) + '-' + str(month_end) + '-' + str(month_range[1])
                        if '日' in i or '天' in i:
                            num = re.search(
                                r'(\d{1,2})[日|天]', i, re.S).group(1)
                            day_end = (datetime.datetime.now() + datetime.timedelta(days=int(num))).day
                            day_start = (datetime.datetime.now() + datetime.timedelta(days=1)).day
                            month_end = (datetime.datetime.now() + datetime.timedelta(days=int(num))).month
                            month_start = (datetime.datetime.now() + datetime.timedelta(days=1)).month
                            time_convert = str(years) + '-' + str(month_start) + '-' + str(
                                day_start) + '~' + str(years) + '-' + str(month_end) + '-' + str(day_end)
                    convert_result.append(time_convert)
            return convert_result
        else:
            return None

    def _infer_time_anglicize(self, context):
        """
        口语型时间映射
        :param context:
        :return: <list of time entity>
        """
        result = self._extract_time(context)
        if result is not None:
            convert_result2 = []
            for i in result:
                i = self._num_transform(i)
                time_convert = ''
                if self._has_numbers(i) is False and (
                        '周' in i or '星期' in i or '礼拜' in i or '季' in i or
                        '年' in i or '天' in i or '月' in i or '最新' in i):
                    '''天的转换'''
                    if i == '今天':
                        time_convert = (
                            datetime.datetime.now()).strftime('%Y-%m-%d')
                    if i == '明天':
                        time_convert = (
                            datetime.datetime.now() +
                            datetime.timedelta(
                                days=1)).strftime('%Y-%m-%d')
                    if i == '昨天':
                        time_convert = (
                            datetime.datetime.now() +
                            datetime.timedelta(
                                days=-
                                1)).strftime('%Y-%m-%d')
                    if i == '后天':
                        time_convert = (
                            datetime.datetime.now() +
                            datetime.timedelta(
                                days=2)).strftime('%Y-%m-%d')
                    if i == '前天':
                        time_convert = (
                            datetime.datetime.now() +
                            datetime.timedelta(
                                days=-
                                2)).strftime('%Y-%m-%d')
                    if '最新' in i:
                        years = str(datetime.datetime.now().year)
                        start_months = datetime.datetime.now().month-1
                        end_months = datetime.datetime.now().month
                        if start_months < 10 and len(str(start_months)):
                            start_months = '0'+str(start_months)
                        else:
                            start_months = start_months
                        if end_months < 10 and len(str(end_months)):
                            end_months = '0' + str(end_months)
                        else:
                            end_months = start_months
                        days = datetime.datetime.now().day
                        if days < 10 and len(str(days)):
                            days = '0' + str(days)
                        else:
                            days = days
                        time_convert = years + '-' + start_months + \
                            '-01' + '~' + years + '-' + end_months + '-' + str(days)
                    '''年的转换'''
                    if i == '今年':
                        month_range = calendar.monthrange(
                            datetime.datetime.now().year, 12)
                        time_convert = str(datetime.datetime.now().year) + '-01-01' + '~' + str(
                            datetime.datetime.now().year) + '-12' + '-' + str(month_range[1])
                    if i == '明年':
                        month_range = calendar.monthrange(
                            datetime.datetime.now().year + 1, 12)
                        time_convert = str(datetime.datetime.now().year + 1) + '-01-01' + '~' + str(
                            datetime.datetime.now().year + 1) + '-12' + '-' + str(month_range[1])
                    if i == '前年':
                        month_range = calendar.monthrange(
                            datetime.datetime.now().year - 2, 12)
                        time_convert = str(datetime.datetime.now().year - 2) + '-01-01' + '~' + str(
                            datetime.datetime.now().year - 2) + '-12' + '-' + str(month_range[1])
                    if i == '后年':
                        month_range = calendar.monthrange(
                            datetime.datetime.now().year + 2, 12)
                        time_convert = str(datetime.datetime.now().year + 2) + '-01-01' + '~' + str(
                            datetime.datetime.now().year + 2) + '-12' + '-' + str(month_range[1])
                    if i == '去年':
                        month_range = calendar.monthrange(
                            datetime.datetime.now().year - 1, 12)
                        time_convert = str(datetime.datetime.now().year - 1) + '-01-01' + '~' + str(
                            datetime.datetime.now().year - 1) + '-12' + '-' + str(month_range[1])

                    '''组合词汇'''
                    if ('去年' in i or '明年' in i) and '今天' in i:
                        months = str(datetime.datetime.now().month)
                        days = str(datetime.datetime.now().day)
                        if '去年' in i:
                            years = str(datetime.datetime.now().year - 1)
                            time_convert = years + '-' + months + '-' + days
                        if '明年' in i:
                            years = str(datetime.datetime.now().year + 1)
                            time_convert = years + '-' + months + '-' + days

                    '''月的转换'''
                    if '月' in i:
                        years = str(datetime.datetime.now().year)
                        if '上' in i:
                            months = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=-
                                    30)).month
                            if months < 10 and len(str(months)) == 1:
                                months = '0' + str(months)
                            else:
                                months = months
                            month_range = calendar.monthrange(
                                int(years), int(months))
                            days = ['01', str(month_range[1])]
                            res = [
                                years +
                                '-' +
                                str(months) +
                                '-' +
                                day for day in days]
                            time_convert = res[0] + '~' + res[1]
                        if '本' in i or '这' in i:
                            months = datetime.datetime.now().month
                            if months < 10 and len(str(months)) == 1:
                                months = '0' + str(months)
                            else:
                                months = months
                            month_range = calendar.monthrange(
                                int(years), int(months))
                            days = ['01', str(month_range[1])]
                            res = [
                                years +
                                '-' +
                                str(months) +
                                '-' +
                                day for day in days]
                            time_convert = res[0] + '~' + res[1]
                        if '下' in i:
                            months = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=30)).month
                            if months < 10 and len(str(months)) == 1:
                                months = '0' + str(months)
                            else:
                                months = months
                            month_range = calendar.monthrange(
                                int(years), int(months))
                            days = ['01', str(month_range[1])]
                            res = [
                                years +
                                '-' +
                                str(months) +
                                '-' +
                                day for day in days]
                            time_convert = res[0] + '~' + res[1]

                    '''周的转换'''
                    if '周' in i or '星期' in i or '礼拜' in i:
                        if "本" in i or "这" in i:
                            week_day = int(
                                datetime.datetime.now().isoweekday())
                            start_time = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=-
                                    week_day +
                                    1)).strftime('%Y-%m-%d')
                            end_time = (
                                datetime.datetime.strptime(
                                    start_time,
                                    '%Y-%m-%d') +
                                datetime.timedelta(
                                    days=6)).strftime('%Y-%m-%d')
                            time_convert = start_time + '~' + end_time
                        if '上' in i:
                            week_day = int(
                                datetime.datetime.now().isoweekday())
                            start_time = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=-
                                    week_day)).strftime('%Y-%m-%d')
                            end_time = (
                                datetime.datetime.strptime(
                                    start_time,
                                    '%Y-%m-%d') +
                                datetime.timedelta(
                                    days=-
                                    6)).strftime('%Y-%m-%d')
                            time_convert = end_time + '~' + start_time
                        if '下' in i:
                            week_day = int(datetime.datetime.now().weekday())
                            d_day = 7 - week_day
                            start_time = (
                                datetime.datetime.now() +
                                datetime.timedelta(
                                    days=d_day +
                                    1)).strftime('%Y-%m-%d')
                            end_time = (
                                datetime.datetime.strptime(
                                    start_time,
                                    '%Y-%m-%d') +
                                datetime.timedelta(
                                    days=6)).strftime('%Y-%m-%d')
                            time_convert = start_time + '~' + end_time
                    '''季度的转换'''
                    if '季度' in i:
                        if self._has_numbers(i) is False:
                            if '本' in i or '这' in i:
                                years = str(datetime.datetime.now().year)
                                month_now = datetime.datetime.now().month
                                mod_cal = month_now % 3
                                if mod_cal == 0:
                                    res1 = [str(month_now - i)
                                            for i in range(3)]
                                    res2 = [
                                        '0' + x for x in res1 if len(x) == 1]
                                    res3 = [years + '-' + x for x in res2]
                                    month_range = calendar.monthrange(
                                        int(years), int(month_now))
                                    days = [str(month_range[1])]
                                    time_convert = res3[2] + '-01' + \
                                        '~' + res3[0] + '-' + days[0]
                                if mod_cal == 1:
                                    res1 = [str(month_now + i)
                                            for i in range(3)]
                                    res2 = [
                                        '0' + x for x in res1 if len(x) == 1]
                                    res3 = [years + '-' + x for x in res2]
                                    month_range = calendar.monthrange(
                                        int(years), int(res1[2]))
                                    days = [str(month_range[1])]
                                    time_convert = res3[0] + '-01' + \
                                        '~' + res3[2] + '-' + days[0]
                                if mod_cal == 2:
                                    res1 = [
                                        str(month_now - 1), str(month_now), str(month_now + 1)]
                                    res2 = [years + '-' + x for x in res1]
                                    month_range = calendar.monthrange(
                                        int(years), int(month_now + 1))
                                    days = str(month_range[1])
                                    time_convert = res2[0] + '-01' + \
                                        '~' + res2[2] + '-' + days
                            if '上' in i:
                                years = str(datetime.datetime.now().year)
                                month_now = datetime.datetime.now().month
                                if month_now > 3:
                                    if int(month_now / 3) <= 1:
                                        time_convert = years + '-01-01' + '~' + years + '-03-31'
                                    if 1 < int(month_now / 3) <= 2:
                                        time_convert = years + '-04-01' + '~' + years + '-06-30'
                                    if 2 < int(month_now / 3) <= 3:
                                        time_convert = years + '-07-01' + '~' + years + '-09-30'
                                else:
                                    print('当前时间为本年度第一季度')
                            if '下' in i:
                                years = str(datetime.datetime.now().year)
                                month_now = datetime.datetime.now().month
                                if month_now > 3:
                                    if int(month_now / 3) <= 1:
                                        time_convert = years + '-04-01' + '~' + years + '-06-30'
                                    if 1 < int(month_now / 3) <= 2:
                                        time_convert = years + '-07-01' + '~' + years + '-09-30'
                                    if 2 < int(month_now / 3) <= 3:
                                        time_convert = years + '-10-01' + '~' + years + '-12-31'
                                else:
                                    print('当前时间为本年度第四季度')
                    convert_result2.append(time_convert)
            return convert_result2
        else:
            return None

    def _infer_time_contact(self, context):
        """判断时间实体之间的关系，确定输出结果为时间区间还是时间点"""
        result = self._extract_time(context)
        standard = self._infer_time_standard(context)
        mixed = self._infer_time_mixed(context)
        anglicize = self._infer_time_anglicize(context)
        convert_result4 = []
        time_convert = ''
        if result is not None:
            if len(result) == 2 and (
                    len(standard) == 2 or len(mixed) == 2 or len(anglicize) == 2):
                if len(standard) == 2:
                    standard = standard
                if len(mixed) == 2:
                    standard = mixed
                if len(anglicize) == 2:
                    standard = anglicize
                patterns = (result[0] + '(.*?)' + result[1])
                link = re.search(patterns, context['query'], re.S).group(1)
                if '到' in link or '至' in link or link == '~' or link == '-':
                    if result[1][0:4].isdigit() is False and '今年' not in result[1] and '去年' not in result[1] and '明年' \
                            not in result[1] and '后年' not in result[1] and '前年' not in result[1]:
                        years = result[0][0:4]
                        months_start = standard[0].split('-')[1]
                        days = standard[1].split('-')[2]
                        if '~' in standard[0] and '~' not in standard[1]:
                            months_end = standard[1].split('-')[1]
                            time_convert = years + '-' + months_start + '-01' + \
                                '~' + years + '-' + months_end + '-' + days
                        elif '~' in standard[0] and '~' in standard[1]:
                            months_end = standard[1].split(
                                '~')[1].split('-')[1]
                            month_range = calendar.monthrange(
                                int(years), int(months_end[1]))
                            time_convert = years + '-' + months_start + '-01' + '~' + \
                                years + '-' + months_end + '-' + str(month_range[1])
                        elif '~' not in standard[0] and '~' not in standard[1]:
                            time_convert = standard[0] + '~' + standard[1]
                        elif '~' not in standard[0] and '~' in standard[1]:
                            months_end = standard[1].split(
                                '~')[1].split('-')[1]
                            month_range = calendar.monthrange(
                                int(years), int(months_end[1]))
                            time_convert = standard[0] + '~' + years + \
                                '-' + months_end + '-' + str(month_range[1])
                    else:
                        if '~' in standard[0]:
                            start_time = standard[0].split('~')[0]
                        else:
                            start_time = standard[0]
                        if '~' in standard[1]:
                            end_time = standard[1].split('~')[1]
                        else:
                            end_time = standard[1]
                        time_convert = start_time + '~' + end_time
                    convert_result4.append(time_convert)
            return convert_result4
        else:
            return None

    @staticmethod
    def _infer_location_entity(context):
        area1_list = [
            '北京',
            '天津',
            '河北',
            '山西',
            '内蒙古',
            '辽宁',
            '吉林',
            '黑龙江',
            '上海',
            '江苏',
            '浙江',
            '安徽',
            '福建',
            '江西',
            '山东',
            '河南',
            '湖北',
            '湖南',
            '广东',
            '广西',
            '海南',
            '重庆',
            '四川',
            '贵州',
            '云南',
            '西藏',
            '陕西',
            '甘肃',
            '青海',
            '宁夏',
            '新疆']
        area2_list = ['石家庄', '唐山', '秦皇岛', '邯郸', '邢台', '保定', '张家口', '承德', '沧州', '廊坊', '衡水',
                      '太原', '大同', '阳泉', '长治', '晋城', '朔州', '晋中', '运城', '忻州', '临汾',
                      '吕梁', '呼和浩特', '包头', '乌海', '赤峰', '通辽', '鄂尔多斯', '呼伦贝尔', '巴彦淖尔',
                      '乌兰察布', '兴安', '锡林郭勒', '阿拉善', '沈阳', '大连', '鞍山', '抚顺', '本溪', '丹东',
                      '锦州', '营口', '阜新', '辽阳', '盘锦', '铁岭', '朝阳', '葫芦岛', '长春', '吉林', '四平',
                      '辽源', '通化', '白山', '松原', '白城', '延边', '哈尔滨', '齐齐哈尔', '鸡西', '鹤岗', '双鸭山',
                      '大庆', '伊春', '佳木斯', '七台河', '牡丹江', '黑河', '绥化', '大兴安岭', '南京', '无锡', '徐州',
                      '常州', '苏州', '南通', '连云港', '淮安', '盐城', '扬州', '镇江', '泰州', '宿迁', '杭州', '宁波',
                      '温州', '嘉兴', '湖州', '绍兴', '金华', '衢州', '舟山', '台州', '丽水', '合肥', '芜湖', '蚌埠',
                      '淮南', '马鞍山', '淮北', '铜陵', '安庆', '黄山', '滁州', '阜阳', '宿州', '六安', '亳州', '池州',
                      '宣城', '福州', '厦门', '莆田', '三明', '泉州', '漳州', '南平', '龙岩', '宁德', '南昌', '景德镇',
                      '萍乡', '九江', '新余', '鹰潭', '赣州', '吉安', '宜春', '抚州', '上饶', '济南', '青岛', '淄博',
                      '枣庄', '东营', '烟台', '潍坊', '济宁', '泰安', '威海', '日照', '莱芜', '临沂', '德州', '聊城',
                      '滨州', '菏泽', '郑州', '开封', '洛阳', '平顶山', '安阳', '鹤壁', '新乡', '焦作', '濮阳', '许昌',
                      '漯河', '三门峡', '南阳', '商丘', '信阳', '周口', '驻马店', '直辖县', '武汉', '黄石', '十堰',
                      '宜昌', '襄阳', '鄂州', '荆门', '孝感', '荆州', '黄冈', '咸宁', '随州', '恩施', '长沙', '株洲',
                      '湘潭', '衡阳', '邵阳', '岳阳', '常德', '张家界', '益阳', '郴州', '永州', '怀化', '娄底', '湘西',
                      '广州', '韶关', '深圳', '珠海', '汕头', '佛山', '江门', '湛江', '茂名', '肇庆', '惠州', '梅州',
                      '汕尾', '河源', '阳江', '清远', '东莞', '中山', '潮州', '揭阳', '云浮', '南宁', '柳州', '桂林',
                      '梧州', '北海', '防城港', '钦州', '贵港', '玉林', '百色', '贺州', '河池', '来宾', '崇左', '海口',
                      '三亚', '三沙', '儋州', '县', '成都', '自贡', '攀枝花', '泸州', '德阳', '绵阳', '广元', '遂宁',
                      '内江', '乐山', '南充', '眉山', '宜宾', '广安', '达州', '雅安', '巴中', '资阳', '阿坝', '甘孜', '凉山',
                      '贵阳', '六盘水', '遵义', '安顺', '毕节', '铜仁', '黔西南', '黔东南', '黔南', '昆明', '曲靖', '玉溪',
                      '保山', '昭通', '丽江', '普洱', '临沧', '楚雄', '红河', '文山', '西双版纳', '大理', '德宏', '怒江',
                      '迪庆', '拉萨', '日喀则', '昌都', '林芝', '山南', '那曲', '阿里', '西安', '铜川', '宝鸡', '咸阳',
                      '渭南', '延安', '汉中', '榆林', '安康', '商洛', '兰州', '嘉峪关', '金昌', '白银', '天水', '武威', '张掖',
                      '平凉', '酒泉', '庆阳', '定西', '陇南', '临夏', '甘南', '西宁', '海东', '海北', '黄南', '海南', '果洛',
                      '玉树', '海西', '银川', '石嘴山', '吴忠', '固原', '中卫', '乌鲁木齐', '克拉玛依', '吐鲁番', '哈密',
                      '昌吉', '博尔塔拉', '巴音郭楞', '阿克苏', '克州', '喀什', '和田', '伊犁', '塔城', '阿勒泰']
        result = []
        location = {}
        for i in area1_list:
            if i in context['query']:
                location['province'] = i
        for i in area2_list:
            if i in context['query']:
                location['city'] = i
        result.append(location)
        if len(result[0]) != 0:
            return result
        else:
            return None


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

    '''测试test文件'''
    for line in open("test.txt", 'r'):
        contexts = dict()
        contexts['query'] = line.split(' ')[0]
        print(contexts)
        a = NerRuleV1()
        b = a.extract(contexts)
        print("b", b)
        print(type(b))
        # if str(b['TimeInterval']).replace(', ', ',') == line.split(' ')[1].strip('\n'):
        #     print('True', '\n')
        # else:
        #     print('False', '\n')
        d = a.transform(contexts)
        print(d, '\n')
        print(type(d))

    # '''单独调试'''
    # contexts = {'query': '未来三天用电量'}
    # a = NerRuleV1()
    # b = a.extract(contexts)
    # print(b)
    # c = a.transform(contexts)
    # print(c)
    # d = a._infer_time_minxed(contexts)
    # print(d)
    # e = a._infer_time_anglicize(contexts)
    # print(e)
    # f = a._infer_time_standard(contexts)
    # print(f)
    # g = a._infer_time_entity(contexts)
    # print (g)
    # h = a.transform(contexts)
    # print(h)
