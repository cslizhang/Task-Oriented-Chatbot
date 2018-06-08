# coding:utf8
# @Time    : 18-6-6 上午9:37
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import datetime as dt
from chatbot.preprocessing.text import cut
from chatbot.core.context import Context
from threading import Timer


class Bot(object):
    def __init__(self, intent_model, ner_model, intent2skills, intent_rule=None):
        """

        :param word2vec_path:
        :param intent_model_path:
        :param intent2skills: <dict>, mapping intent_idx to Class skill
        :param interface: <class Interface>
        :param default_skill: <class Skill>
        """
        self.contexts = dict()
        self.intent2skills = intent2skills
        self.intent_model = intent_model
        self.ner_model = ner_model
        skills = []
        for i, s in self.intent2skills.item():
            if s not in skills:
                skills.append(s)
        self.skills = skills
        self.intent_rule = intent_rule
        self._recover_timeout_context()

    @property
    def skill2slots(self):
        return {skill.name: skill.init_slots for skill in self.skills}

    def _intent_recognition(self, context_id):
        def _get_last_intent(intents):
            if len(intents) == 0:
                return None
            else:
                return intents[-1]
        model_intent, m_confidence = self.intent_model.predict(self.contexts[context_id]["query_cut"])
        rule_intent, r_confidence = self.intent_rule(self.contexts[context_id]) \
            if self.intent_rule else (None, None)
        # 上下文进行判断的规则
        last_intent = _get_last_intent(self.contexts[context_id]["intent"])
        if rule_intent is not None:
            return rule_intent, r_confidence
        if (m_confidence <= 0.7) and (self.intent2skills[last_intent].check_update_slots(
            self.contexts[context_id]["entities"]
        )):
            return last_intent, 1.
        return model_intent, m_confidence

    def chat(self, query):
        """

        :param query: <dict>
        :return:
        """
        user = query["user"]
        app = query["app"]
        right = query["right"]
        text = query["text"]
        context_id = self._get_context_id(user, app, right)
        # 检查是否已有对应上下文，没有则新建
        if context_id not in self.contexts.keys():
            self.contexts[context_id] = Context(user, app, self.skill2slots, context_id, right)
        context = self.contexts[context_id]
        # NER
        context["entities"] = self.ner_model.predict(context["query"])
        # 根据信息更新上下文信息
        self._update_context(context_id, text)
        # 意图识别(模型+规则+上下文)
        intent, confidence = self._intent_recognition(context_id)
        context["intent"] = intent
        context["history_intent"].append(intent)
        # 更新词槽
        self._update_slot(context_id, intent)
        # 给出回复
        result = self.intent2skills[intent](self.contexts[context_id])
        return result

    def _update_slot(self, context_id, intent):
        skill = self.intent2skills[intent]
        for slot_name, slot_value in self.contexts[context_id]["entities"].items():
            try:
                self.contexts[context_id]["slots"][skill][slot_name] = slot_value
            except:
                # TODO
                pass

    @staticmethod
    def _get_context_id(user, app, right=None):
        context_id = str(hash(user + app + str(right)))
        return context_id

    def _recover_timeout_context(self):
        new_contexts = dict()
        for context_id, context in self.contexts.items():
            if context.is_timeout:
                pass
            else:
                new_contexts[context_id] = context
        self.contexts = new_contexts
        timer = Timer(1, self._recover_timeout_context)
        timer.start()

    def _update_context(self, context_id, text):
        text_cut = cut(text,)
        self.contexts[context_id]["last_query_time"] = dt.datetime.now()
        self.contexts[context_id]["query_cut"] = text_cut
        self.contexts[context_id]["query"] = text
        self.contexts[context_id]["history_query"].append(text)


if __name__ == "__main__":
    pass
