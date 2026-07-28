# CropGenome-FM研究指南（v2.1白话说明版）

本目录是CropGenome-FM当前机器证据与下一阶段研究设计的公开交付包。

- [详细中文Markdown报告](README_CN.md)
- [详细中文Word报告](CropGenome-FM_详细研究设计与评估报告_CN.docx)
- [汇总数据表说明](source_data/README.md)
- [配套图](figures/)

报告严格区分：已经完成的正式机器结果、只用于过程检查的结果，以及尚未执行的未来任务设计。NT-v2 500M目前是下一版必加基线，尚无本项目正式成绩；64K只训练到第569次参数更新，不能视为已完成模型。

v2.1在v2.0证据内容不变的基础上，重点改写了阅读方式：

- 新增“10步看懂我们具体怎么做”的白话流程；
- 首次出现的shard、window、token、checkpoint、probe等术语全部先用中文解释；
- 数据抽样、模型结构、训练停止、下游比较、未来任务和统计规则均改为“先讲实际做法，再保留技术名”；
- 没有新增或重跑正式测试，所有性能数字与v2.0一致。

v2.0已经新增：

- 核心3任务和外部7任务的科、属、种与Stage B预训练交叉审计；
- 本模型相对AgroNT 1B、PlantCAD2 Small和PlantCaduceus的当前差距表；
- 9项作物专属复杂任务逐项对应AgroNT 1B、PlantCAD2/PlantCaduceus、NT-v2 500M、Evo2及传统强基线的公平比较角色和预注册成功门槛；
- 所有新增机器表均可由`scripts/build_evidence_update_tables.py`重建。
