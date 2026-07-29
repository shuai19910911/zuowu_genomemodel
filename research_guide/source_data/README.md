# Source data说明

这些TSV是从本地冻结manifest、正式summary和prediction中聚合的可公开小表，不包含FASTA/GFF、checkpoint、embedding、主机地址或私有凭据。

| 文件 | 内容 |
|---|---|
| `assembly_species_summary.tsv` | 258个NCBI assembly按30物种汇总的split、来源和压缩文件字节数 |
| `pretraining_stage_summary.tsv` | Stage B/C1/C2/D的窗口、token、context和执行状态 |
| `stage_b_shard_sampling_audit.tsv` | 42个Stage B分片逐项记录碱基数、总/训练/验证/测试片段数、4K/8K/16K组成及旧读取器下单条训练片段的相对抽样权重 |
| `model_architecture_parameters.tsv` | 当前369,505,287参数模型的真实实现参数 |
| `pretraining_objectives.tsv` | 当前已实现与下一版建议的预训练目标，带状态字段 |
| `core_task_summary.tsv` | 核心3任务的样本、物种split和hard-negative定义 |
| `core_primary_metrics.tsv` | 核心任务full-data正式AUPRC；保留锁定formal evaluator原值 |
| `core_auprc_tie_audit_summary.tsv` | 对核心prediction重算的tie-safe Average Precision审计；主学习模型排名不变 |
| `external_task_summary.tsv` | 外部7任务的样本、维度、长度、去重和validation策略 |
| `external_primary_metrics.tsv` | 外部7任务的primary-seed full-data主指标 |
| `aggregate_primary_metrics.tsv` | 核心/外部任务组宏平均，含seed口径 |
| `baseline_registry.tsv` | 已评估和下一版必加基线的参数、架构、上下文与状态 |
| `baseline_gap_summary.tsv` | 从正式聚合指标自动计算的本模型与当前最强植物/公开基线差距；NT-v2 500M仍标记为未评估 |
| `downstream_taxonomy_detail.tsv` | 71行逐面板×任务×split×物种的科属种、样本数、核心assembly和Stage B预训练交叉审计 |
| `downstream_taxonomy_summary.tsv` | 核心3任务＋外部7任务的物种/属/科互斥、test accession隔离和预训练未见比例汇总 |
| `future_task_registry.tsv` | 9项作物专属任务的预注册式设计；计划样本数不是现有样本数 |
| `future_task_baseline_capability_matrix.tsv` | 9项任务逐项说明AgroNT 1B、PlantCAD2/PlantCaduceus、NT-v2 500M、Evo2和传统强基线的公平角色、预计压力点及成功门槛；全部属于设计而非结果 |
| `public_resource_registry.tsv` | 公开数据/模型URL和冻结revision |

重要：旧核心formal evaluator对完全相同prediction score的AUPRC tie处理不正确，因此`majority_baseline`原值0.30734不应作科学解释；标准Average Precision为0.5。正式学习模型排名经重算不变。下一release需要修正实现并重建全部正式artifact。

新增四表由`../scripts/build_evidence_update_tables.py`从真实样本表、258条canonical assembly manifest、现有聚合指标和冻结未来任务registry生成。分类学审计区分下游监督split隔离、Stage B assembly隔离、预训练未见物种、未见属和未见科；不得将这些概念合并成一句“跨物种零样本”。未来任务表中的基线“预计限制”是待检验假设，不是已经观察到的性能失败。

`stage_b_shard_sampling_audit.tsv`由`../scripts/build_stage_b_shard_sampling_audit.py`逐个读取本地冻结Stage B窗口索引生成。脚本验证42个压缩索引的SHA-256、manifest片段数和按4K/8K/16K重算的碱基总数，再按真实旧读取器公式计算抽样权重。公开表不包含DNA序列；原始约41.24 GB碱基编码和窗口索引仍不提交到GitHub。
