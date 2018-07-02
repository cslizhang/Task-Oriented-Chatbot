# Task-Oriented-Chatbot

A framework for Task-Oriented-Chatbot.

## TODO

1. 实体提取（location提取模块，时间提取模块优化{最近、最新}）
2. 数据查询skill(时间、项目名、权限、指标)
3. 政策文件（数据源、清洗规则、标签集合、update机制）
4. 政策查询skill优化
5. 留言skill
6. 对话记录模块
7. 记录打分skill（scoer goodbye）& dm 
8. 公司咨询skill-fengge
9. 业务查询skill-fengge
10. 名词解释skill
11. 项目日志模块优化
12. 语料更新、添加
13. 框架优化
14. 自我迭代机制嵌入
15. 无标签语聊、分类语聊、ner语聊、rank语料、回复逻辑语聊、政策语聊

## feature

+ 将推荐系统融入chatbot，在连接进入时候推荐相关咨询，在意图完毕后，推荐相关后续他问题等。
+ 将上下文融入意图识别模型，rnn，用上论会话的hidden state作为下论会话的init hidden state

## Skill
+ 数据查询  

项目用电相关数据查询，支持查询维度：项目、时间、权限、指标{度电成本、用电总量等}、维度{min max avg}

+ 文件检索  

  + 政策文件
  + 政策研究部门报告
  + 名词解释
  + 多轮政策查询
  + 政策摘要、精准定位
  + 政策问答

+ 公司咨询

  
## Mongodb

梁量组织一次分享培训；根本据chatbot需求确定db方案。
  


## References

+ [Chatbot Framework]()
+ [Text Classification](./docs/references/TextClassification.md)
+ [Named Entity Recognition]()
+ [Dialog Management]()
+ [Parameter Optimization](./docs/references/ParameterOptimization.md)