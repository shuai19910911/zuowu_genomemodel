# CropGenome-FM研究指南（v2.0）

本目录是CropGenome-FM当前机器证据与下一阶段研究设计的公开交付包。

- [详细中文Markdown报告](README_CN.md)
- [详细中文Word报告](CropGenome-FM_详细研究设计与评估报告_CN.docx)
- [聚合source-data说明](source_data/README.md)
- [配套图](figures/)

报告严格区分：已经完成的正式机器结果、仅用于诊断的结果，以及尚未执行的未来任务设计。NT-v2 500M目前是下一版必加基线，尚无本项目正式成绩；Stage C1 64K只训练到step569，不能视为已完成模型。

v2.0新增：

- 核心3任务和外部7任务的科、属、种与Stage B预训练交叉审计；
- 本模型相对AgroNT 1B、PlantCAD2 Small和PlantCaduceus的当前差距表；
- 9项作物专属复杂任务逐项对应AgroNT 1B、PlantCAD2/PlantCaduceus、NT-v2 500M、Evo2及传统强基线的公平比较角色和预注册成功门槛；
- 所有新增机器表均可由`scripts/build_evidence_update_tables.py`重建。
