# ChatBot设计稿

## 关键概念

+ Session

一个完整的多轮对话单元
 
+ Context
 
语境上下文，有公共语境和意图语境

+ Intent

意图，意图分类器(IntentClassifier)+意图规则输出的结果

+ Entity

实体，包含时间、地点、组织、业态、设备等多种实体，是实体提取器(NameEntityRecognizer)的输出结果

+ Slot

回复API所需的参数，视为槽，槽有多种类型（是否必须取值）、取值和更新方法

+ Message

Query和Response的父类，封装消息内容，消息渠道，消息状态，消息发送人等信息

+ Action

行动类，有多种状态，表示用户在该行动下所处的位置，如果所有条件均满足，返回查询结果，否则是多轮会话状态
行动类和意图为11映射关系

+ User

用户类，用户id，用户名字等。。

+ LittleChild

机器人自己

+ interface

服务接口，微信，APP，WEB等

## 一个标准服务流程

### 1）初始化

初始化一个bot，定义bot服务的interface，webAPI相关属性

### 2）一个新的