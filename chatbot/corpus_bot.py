# coding:utf8
# @Time    : 18-6-2 下午12:06
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
def insert(text, name):
    path = "/home/zhouzr/project/Task-Oriented-Chatbot/corpus/intent_corpus.txt"
    text = text.replace("\u2005", " ")
    text = text.replace("  ", " ")
    print(text.split(" "))
    t = text.split(" ")
    flag = False
    if len(t) == 3:
        with open(path, 'a') as f:
            s = " ".join([name, t[1], t[2], "\n"])
            print(s)
            f.write(s)
            flag = True
    return flag


if __name__ == "__main__":
    from wxpy.utils.misc import enhance_connection
    import requests

    URL = "http://www.tuling123.com/openapi/api"
    APIKEY = "7e3bf5d6d06143e39c898672592d63ad"
    USERID = "test"
    LOC = "成都"
    sesstion = requests.session()
    enhance_connection(sesstion)


    def tuling(s):
        send = dict(
            key=APIKEY,
            userid=USERID,
            info=s,
            loc=LOC
        )

        r = sesstion.post(URL, send)
        answer = r.json()
        return answer["text"]

    from wxpy import *
    # 初始化机器人，扫码登陆
    bot = Bot()
    my_friend = bot.friends()
    my_groups = bot.groups()


    @bot.register(my_friend)
    def reply_my_friend(msg):
    #     print(type(msg.text))
    #     return '正在测试chatbot\n输入文本:{}\n输入类型:{}\n对话id:{}\n昵称:{}'.format(
    #         msg.text,
    #         msg.type,
    #         msg.id,
    #         msg.sender.remark_name,

    #     )
        pass


    @bot.register(my_groups)
    def reply_my_group(msg):
        if msg.is_at:
            if "今天" in msg.text:
                return """你好，chatbot需要你的帮助，优化意图识别模型\n
                目前支持需要完善的意图有：公司咨询（地址联系方式等）、业务咨询（产品、业务范围）、数据查询（用电数据）、文件检索（政策文件）。\n
                例子：\n
                @周知瑞 公司咨询 你们公司地址在哪？\n
                @周知瑞 业务咨询 你们有什么服务\n
                @周知瑞 数据查询 上个星期3到星期2，来福士空调用了多少电？\n
                @周知瑞 文件检索 最近一个月四川能监办，跨省跨区文件\n
                """

            elif insert(msg.text, msg.member.name):
                return "谢谢{},成功添加一条意图识别样本".format(msg.member.name)

            else:
                text = " ".join(msg.text.replace("\u2005", " ").split(" ")[1:])
                return tuling(text)

    bot.join()