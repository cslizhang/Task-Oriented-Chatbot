# coding:utf8
# @Time    : 18-6-6 上午10:02
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import datetime as dt
from sklearn.externals.joblib import load
import pandas as pd

from chatbot.core.discarded.base_skill import BaseSkill


class FileRetrieval(BaseSkill):
    def __init__(self, tfidf_path, cluster_index_path, file_path, k=5, limit_distance=0.99):
        super().__init__()
        self._tfidf = load(tfidf_path)
        self._ci = load(cluster_index_path)
        self._file = pd.read_csv(file_path, parse_dates=[0])
        self._k = k
        self._limit_distance = limit_distance

    @property
    def name(self):
        return "file_retrieval"

    @property
    def init_slots(self):
        now = dt.datetime.now().date()
        return {
            "location":
                {"province": None, "city": None},
            "time_interval":
                {
                    "start": str(now.replace(year=now.year-1)),
                    "end": str(now)
                }
        }

    def __call__(self, context):
        """ 如果满足调用条件，返回回复，否则，返回对应的问题

        :param context:
        :return:
        """
        is_satisfied, question = self._check_satisfied(context)
        if is_satisfied:
            return self._act(context)
        else:
            return question

    def check_update_slots(self, entities):
        for k, v in entities.items():
            if k in self.init_slots.keys():
                return True
        return False

    def _check_satisfied(self, context):
        return True, None

    def _act(self, context):
        q_tfidf = self._tfidf.transform([context["text_cut"]]).toarray()
        search_result = self._ci.search(q_tfidf, k=self._k)[0]
        slot = context["slots"][self.name]
        if len(search_result) == 0:
            return self._not_find
        result = []
        idx_time_limit = self._file[(self._file.date >= slot["time_interval"]["start"]) & \
                               (self._file.date <= slot["time_interval"]["end"])].index.tolist()
        idx_location_limit = self._file.index.tolist() if slot["location"]["province"] is None \
            else self._file[(self._file.area==slot["location"]["province"])].index.tolist()
        idx_limit = set(idx_location_limit) & set(idx_time_limit)
        for (d, i) in search_result:
            if (d <= self._limit_distance) and (i in idx_limit):
                result.append(self._response_template(province=self._file.loc[i, "area"],
                                                      time=self._file.loc[i, "date"].date(),
                                                      name=self._file.loc[i, "name"],
                                                      url=self._file.loc[i, "url"],
                                                      distance=d
                                                      ))
        if len(result) == 0:
            return self._not_find
        else:
            head = self._response_head(slot["location"]["province"],
                                       startdate=slot["time_interval"]["start"],
                                       enddate=slot["time_interval"]["end"]
                                       )
            return head + "\n".join(result)

    @staticmethod
    def _response_template(province, time, name, url, distance):
        return "{} {}：{}，{}，相似度:{:.1f}%".format(province, time, name,
                                             url, (1-distance)*100)

    @staticmethod
    def _response_head(location=None, startdate=None, enddate=None):
        l = "不限" if location is None else location
        s = "不限" if enddate is None else startdate
        e = "不限" if enddate is None else enddate
        return "您查询的地区{}, {}至{}相关文件如下: \n".format(l, s, e)

    @property
    def _not_find(self):
        return "抱歉，您所查找的政策文件不存在，小益已经上报，可能明天就有了哦～"


if __name__ == "__main__":
    skill = FileRetrieval("/home/zhouzr/project/Task-Oriented-Chatbot/chatbot/results/tfidf",
                          "/home/zhouzr/project/Task-Oriented-Chatbot/chatbot/results/cluster_index",
                          "/home/zhouzr/project/Task-Oriented-Chatbot/corpus/policy_file.utf8.csv")

    from chatbot.preprocessing.text import cut

    context = {"text_cut": " ".join(cut("")),
     "slots":{"file_retrieval":
                  skill.init_slots
              }}
    context["slots"]["file_retrieval"]["time_interval"]["end"]="2018-02-02"
    context["slots"]["file_retrieval"]["time_interval"]["start"] = "2017-01-02"
    print(skill(context))
