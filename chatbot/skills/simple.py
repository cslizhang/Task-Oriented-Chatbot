# coding:utf8
# @Time    : 18-6-11 下午2:41
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import random
import codecs

from chatbot.core.skill import BaseSkill
from chatbot.utils.path import ROOT_PATH
from chatbot.utils.log import get_logger


logger = get_logger("simple skill")


def read_txt(path):
        """read txt

        :return: <list>
        """        
#        path = "D:\\Users\\tanmx\\chatbot\\Task-Oriented-Chatbot\\corpus\\skill\\GoodBye_response.txt"
        with open(path, "r", encoding='UTF-8') as f:
            txts = f.readlines()
        # remove chomp, blank
        sents = [item.strip().split(' ')[-1] for item in txts if len(item) > 1]
        return sents


class LeaveMessage(BaseSkill):
    """LeaveMessage存储及回复封装
      :param context: context
      :return: <String> 回复信息,context{user:,query} to txt
      """
    def __init__(self, path=None):
        if path is None:
            self.path = str(ROOT_PATH / "log" / "message")
        else:
            self.path = path
        logger.debug("leave message save in %s" % self.path)

    def __call__(self, context):
        with codecs.open(self.path, "a+", "utf-8") as f:
            f.write(context['context_id'] + '\t' + '\t' + context['app'] + '\t' + '\t' + context[
                'last_query_time'] + '\t' + '\t' + context['user'] + '\t' + '\t' + context['query'] + '\n' + '\n')
        f.close()
        return "小益已经帮您记下啦"

    def contain_slots(self, entities):
        return False

    def init_slots(self):
        return {}


class SayHi(BaseSkill):
    """SayHi回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问
    
    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """    
    
    def __init__(self):
        """initialize
        
        get the corpus' path
        get the corpus
        """
        print("loading response for SayHi...")
        path = str(ROOT_PATH.parent / "corpus"/"intent"/"skill"/"SayHi_response.txt")
        self.reps = read_txt(path)

    def __call__(self, context):
        """skill回复逻辑封装
        
        满足skill需求给出回复
        :param context: <class Context>
        :return: <String> 回复信息
        """
        answer_text = random.choice(self.reps)
        answer_text = answer_text + context["user"]
        return answer_text

    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {}

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        return False


class GoodBye(BaseSkill):
    """GoodBye回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问
    
    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """    
    
    def __init__(self):
        """initialize
        
        get the corpus' path
        get the corpus
        """
        print("loading response for GoodBye...")
        path = str(ROOT_PATH.parent / "corpus"/"intent"/"skill"/"GoodBye_response.txt")
        self.reps = read_txt(path)

    def __call__(self, context):
        """skill回复逻辑封装
        
        满足skill需求给出回复
        :param context: <class Context>
        :return: <String> 回复信息
        """
        answer_text = random.choice(self.reps)
        answer_text = answer_text + context["user"]
        return answer_text

    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {}

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        return False

        
class Thanks(BaseSkill):
    """Thanks回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问
    
    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """    
    
    def __init__(self):
        """initialize
        
        get the corpus' path
        get the corpus
        """
        print("loading response for Thanks...")
        path = str(ROOT_PATH.parent / "corpus"/"intent"/"skill"/"Thanks_response.txt")
        self.reps = read_txt(path)

    def __call__(self, context):
        """skill回复逻辑封装
        
        满足skill需求给出回复
        :param context: <class Context>
        :return: <String> 回复信息
        """
        answer_text = random.choice(self.reps)
        answer_text = answer_text + context["user"]
        return answer_text

    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {}

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        return False


class Praise(BaseSkill):
    """Praise回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问
    
    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """    
    
    def __init__(self):
        """initialize
        
        get the corpus' path
        get the corpus
        """
        print("loading response for Praise...")
        path = str(ROOT_PATH.parent / "corpus"/"intent"/"skill"/"Praise_response.txt")
        self.reps = read_txt(path)

    def __call__(self, context):
        """skill回复逻辑封装
        
        满足skill需求给出回复
        :param context: <class Context>
        :return: <String> 回复信息
        """
        answer_text = random.choice(self.reps)
        answer_text = answer_text + context["user"]
        return answer_text

    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {}

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        return False


class Criticize(BaseSkill):
    """Criticize回复逻辑封装

    满足skill需求给出回复，不满足需求，给出反问
    
    :param context: <class Context> 当前会话上下文
    :return: <String> 回复信息
    """    
    
    def __init__(self):
        """initialize
        
        get the corpus' path
        get the corpus
        """
        print("loading response for Criticize...")
        path = str(ROOT_PATH.parent / "corpus"/"intent"/"skill"/"Criticize_response.txt")
        self.reps = read_txt(path)

    def __call__(self, context):
        """skill回复逻辑封装
        
        满足skill需求给出回复
        :param context: <class Context>
        :return: <String> 回复信息
        """
        answer_text = random.choice(self.reps)
        answer_text = answer_text + context["user"]
        return answer_text
    
    def init_slots(self):
        """skill name

        :return: <dict> slots dict
        """
        return {}

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        return False
    
















