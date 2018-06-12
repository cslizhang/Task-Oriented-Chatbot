# coding:utf8
# @Time    : 18-6-6 上午9:37
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import datetime as dt
from chatbot.preprocessing.text import cut
from chatbot.core.context import Context
from threading import Timer


class Bot(object):
    def __init__(self, vocab, label, intent_model,
                 intent_rule, ner, intent2skills):
        self.contexts = dict()
        self.intent2skills = intent2skills
        self.intent_model = intent_model
        self.intent_rule = intent_rule
        self.ner = ner
        skills = []
        for i, s in self.intent2skills.item():
            if s not in skills:
                skills.append(s)
        self.skills = skills
        self.vocab = vocab
        self.label = label
        self._delete_timeout_context()

    @property
    def skill2slots(self):
        return {skill.name(): skill.init_slots for skill in self.skills}

    def _intent_recognition(self, context_id):
        def _get_last_intent(intents):
            if len(intents) == 0:
                return None
            else:
                return intents[-1]
        model_rst = self.intent_model.infer(self.contexts[context_id]["query_idx"])
        rule_rst = self.intent_rule.infer(self.contexts[context_id]["query"])

        if rule_rst is not None:
            return rule_rst

        # 上下文进行判断的规则
        last_intent = _get_last_intent(self.contexts[context_id]["intent"])
        if last_intent is None:
            return model_rst
        elif (model_rst[1] <= 0.2) and (self.intent2skills[last_intent].contain_slots(
            self.contexts[context_id]["entities"]
        )):
            return last_intent, -1.
        else:
            return model_rst

    def chat(self, query):
        """

        :param query: <dict>
        :return:
        """
        context_id = self._get_context_id(query)
        # 检查是否已有对应上下文，没有则新建
        if context_id not in self.contexts.keys():
            self.contexts[context_id] = Context(user=query["user"], app=query["app"],
                                                skill2slot=self.skill2slots, context_id=context_id, right=query["right"])
        # 根据信息更新上下文信息,文本，分词，映射id，实体，意图
        self._update_context(context_id, query)
        # 更新词槽
        self._update_slot(context_id)
        # 给出回复
        result = self.intent2skills[self.contexts[context_id]["intent"]](self.contexts[context_id])
        return result

    def _update_slot(self, context_id):
        skill = self.intent2skills[self.contexts[context_id]["intent"]]
        for slot_name, slot_value in self.contexts[context_id]["entities"].items():
            try:
                self.contexts[context_id]["slots"][skill][slot_name] = slot_value
            except KeyError:
                # TODO
                pass

    @staticmethod
    def _get_context_id(query):
        user = query["user"]
        app = query["app"]
        right = query["right"]
        context_id = str(hash(user + app + str(right)))
        return context_id

    def _delete_timeout_context(self):
        self.contexts = {k: v for k, v in filter(lambda item: item[1].is_timeout, self.contexts.items())}
        timer = Timer(1, self._delete_timeout_context)
        timer.start()

    def _update_context(self, context_id, query):
        text = query["text"]
        text_cut = cut(text)
        self.contexts[context_id]["last_query_time"] = dt.datetime.now()
        self.contexts[context_id]["query_cut"] = text_cut
        self.contexts[context_id]["query_idx"] = self.vocab.transform_one(text_cut)
        self.contexts[context_id]["query"] = text
        self.contexts[context_id]["history_query"].append(text)
        self.contexts[context_id]["entities"] = self.ner.infer(text)
        intent, confidence = self._intent_recognition(context_id)
        self.contexts["context"]["intent"] = intent
        self.contexts["context"]["history_intent"].append(intent)


if __name__ == "__main__":
    pass
