# Source data说明

这些TSV是从本地冻结manifest、正式summary和prediction中聚合的可公开小表，不包含FASTA/GFF、checkpoint、embedding、主机地址或私有凭据。

| 文件 | 内容 |
|---|---|
| `assembly_species_summary.tsv` | 258个NCBI assembly按30物种汇总的split、来源和压缩文件字节数 |
| `pretraining_stage_summary.tsv` | Stage B/C1/C2/D的窗口、token、context和执行状态 |
| `model_architecture_parameters.tsv` | 当前369,505,287参数模型的真实实现参数 |
| `pretraining_objectives.tsv` | 当前已实现与下一版建议的预训练目标，带状态字段 |
| `core_task_summary.tsv` | 核心3任务的样本、物种split和hard-negative定义 |
| `core_primary_metrics.tsv` | 核心任务full-data正式AUPRC；保留锁定formal evaluator原值 |
| `core_auprc_tie_audit_summary.tsv` | 对核心prediction重算的tie-safe Average Precision审计；主学习模型排名不变 |
| `external_task_summary.tsv` | 外部7任务的样本、维度、长度、去重和validation策略 |
| `external_primary_metrics.tsv` | 外部7任务的primary-seed full-data主指标 |
| `aggregate_primary_metrics.tsv` | 核心/外部任务组宏平均，含seed口径 |
| `baseline_registry.tsv` | 已评估和下一版必加基线的参数、架构、上下文与状态 |
| `future_task_registry.tsv` | 9项作物专属任务的预注册式设计；计划样本数不是现有样本数 |
| `public_resource_registry.tsv` | 公开数据/模型URL和冻结revision |

重要：旧核心formal evaluator对完全相同prediction score的AUPRC tie处理不正确，因此`majority_baseline`原值0.30734不应作科学解释；标准Average Precision为0.5。正式学习模型排名经重算不变。下一release需要修正实现并重建全部正式artifact。
