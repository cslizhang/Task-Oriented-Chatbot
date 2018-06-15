# coding:utf8
# @Time    : 18-6-11 下午2:41
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import random
from chatbot.core.skill import BaseSkill
from chatbot.utils.path import ROOT_PATH
# D:\Users\tanmx\chatbot\Task-Oriented-Chatbot\chatbot
# TODO:tanmx 回复逻辑 & 对应的样本

class LeaveMessage(BaseSkill):
    pass


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
        self.reps = self.read_txt(path)

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
        return dict

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
        self.reps = self.read_txt(path)

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
        return dict

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
        self.reps = self.read_txt(path)

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
        return dict

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
        self.reps = self.read_txt(path)

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
        return dict

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
        self.reps = self.read_txt(path)

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
        return dict

    def contain_slots(self, entities):
        """判断是否包含skill所需词槽

        :param entities: <list of dict>
        :return: <bool> True包含，False不包含
        """
        return False
    
    
#context = {"query": "哈哈", "user": "周"}    
#sayhi = SayHi() 
#goodbye = GoodBye()    
#thanks = Thanks()
#praise = Praise()
#criticize = Criticize()  
# 
#sayhi(context)
#sayhi.init_slots()
#
#goodbye(context)
#
#thanks(context)
#
#praise(context)
#
#criticize(context)
















