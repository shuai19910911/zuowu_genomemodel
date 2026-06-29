# CropGenome-FM 作物基因组基础模型

更新时间: 2026-06-29 08:20 CST

CropGenome-FM（Crop Genome Foundation Model，作物基因组基础模型）是一个面向作物结构注释基因组的 DNA language model（DNA 语言模型）。项目目标不是再做一个通用 DNA 模型，而是用结构注释完整的作物基因组数据训练一个更适合 crop-specific sequence understanding（作物专用序列理解）的长上下文模型，并用独立下游 benchmark（基准评测）证明它在剪接、启动子、终止、基因结构、转座元件边界和跨物种迁移等任务上的价值。

## 1. GitHub 只保留的入口

本仓库的 GitHub 可见内容只保留下面几个入口文档和核心图，避免旧实验结果、临时方案、逐 checkpoint（模型存档点）明细和分散图表干扰判断。

| 入口 | 用途 | 怎么看 |
|---|---|---|
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | 研究详细方案 | 看数据口径、训练分阶段、下游 benchmark（基准评测）、消融和风险控制。 |
| [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) | 模型结构解释 | 看 v2 Stable（第二版稳健版）到底输入什么、主干是什么、loss（损失函数）怎么选、哪些不能过度声明。 |
| [TRAINING_PROGRESS.md](TRAINING_PROGRESS.md) | 唯一训练进展文档 | 所有训练曲线、下游 probe（探针评测）图表、结构注释进度和结论都集中在这里；只看这一个文档即可了解进展。 |
| [assets/cropgenome_fm_roadmap.svg](assets/cropgenome_fm_roadmap.svg) | 研究方案图 | 对 PROJECT_PLAN 的图形化概览。 |
| [assets/cropgenome_fm_model_architecture.svg](assets/cropgenome_fm_model_architecture.svg) | 模型架构图 | 对 MODEL_ARCHITECTURE 的图形化概览。 |

训练输入 shard（分片）、checkpoint（模型存档点）、原始 genome（基因组序列）、GFF/GTF（结构注释文件）、中间索引、逐样本预测和大日志不上传 GitHub。

## 2. 当前研究定位

### 2.1 做什么

- 使用结构注释完整的 crop assembly（作物基因组装版本）构建预训练语料。
- 在 8K context（8192 碱基上下文）上训练 `CropGenome-FM-v2-Stable-8K`（作物基因组基础模型第二版稳健 8192 碱基版）。
- 后续按资源扩展到 64K/128K context（更长基因组上下文）。
- 重点评估 splice donor/acceptor（剪接供体/受体）、promoter/TSS（启动子/转录起始位点）、TES/polyA（转录终止/多聚腺苷酸化）、exon/intron/UTR（外显子/内含子/非翻译区）、TE boundary（转座元件边界）和跨物种迁移。

### 2.2 为什么这样做

通用 DNA 模型通常依赖人类或混合物种数据，作物基因组具有更强的 repeat（重复序列）、TE（转座元件）、多倍化和注释质量差异。直接使用作物结构注释数据，可以让模型在预训练阶段就更多看到 coding（编码区）、splice（剪接区）、promoter（启动子）、UTR（非翻译区）和 gene body（基因体）等功能区域，而不是只学习无差别全基因组背景。

### 2.3 当前评估

当前 v2 Stable（第二版稳健版）已训练到 step2530，step2000 validation selection loss（验证选择损失）降到 1.1890，并更新 `checkpoint_best.pt`（最佳模型存档点）。step1000 lightweight downstream probe（轻量下游探针评测）已完成但只算弱阳性；正式结论仍要等 splice/promoter/TES（剪接/启动子/转录终止）等独立 benchmark（基准评测）。最新状态只看 [TRAINING_PROGRESS.md](TRAINING_PROGRESS.md)。

## 3. 数据和安全边界

| 项目 | 当前口径 |
|---|---|
| 正式训练数据 | 263 行同时有 genome FASTA（基因组序列）和 GFF3/GTF（结构注释）的 crop assembly manifest（作物组装清单） |
| 去重后 assembly accession（组装版本号） | 258 个 canonical assembly accession（标准化组装版本） |
| 覆盖属 | 26 个 |
| train/val/test split（训练/验证/测试划分） | 192/35/31，按 accession 防泄漏 |
| 训练服务器输入目录 | `training_server_transfer/`，只在本地/训练服务器使用，不上传 GitHub |
| GitHub 上传策略 | 只上传入口文档、核心 SVG（矢量图）、少量核心训练 PNG（位图）和必要 TSV（表格源数据）；下游明细目录只本地保留 |

评估: 这个口径牺牲了 1644 个无结构注释 genome（基因组），但换来更干净的区域标签、更可解释的下游任务和更低的数据泄漏风险。第一版论文叙事应强调“结构注释完整作物基因组预训练 + 独立 benchmark（基准评测）”，而不是宣称数据量最大。

## 4. 当前主模型

当前主线是 `CropGenome-FM-v2-Stable-8K`：

- single-base token（单碱基 token），避免 k-mer（固定长度片段）切词影响单点变异解释。
- HyenaLite（轻量长卷积序列模型）+ local attention（局部注意力）主干。
- MLM（masked language modeling，遮盖碱基预测）为主任务。
- RC consistency（reverse-complement consistency，反向互补一致性）作为小权重正则。
- weak region auxiliary head（弱监督区域辅助头）只作训练辅助和 sanity check（健康检查），不作为正式创新证据。
- best checkpoint（最佳模型存档点）和 early stopping（早停）按 `selection_loss = MLM loss + 0.02 * RC loss`（遮盖预测损失 + 小权重反向互补损失）选择。

评估: 这个模型设计偏稳健，不把架构新奇性作为论文主卖点。项目成功与否主要看独立下游 benchmark（基准评测）和跨物种泛化，而不是只看预训练 loss（损失）是否下降。

## 5. 如何判断项目是否在变好

只看 [TRAINING_PROGRESS.md](TRAINING_PROGRESS.md)：

1. v2 Stable（第二版稳健版）train loss（训练损失）是否持续下降。
2. step1000 之后 validation loss（验证损失）和 selection loss（选择损失）是否下降。
3. best checkpoint（最佳模型存档点）是否稳定出现，而不是只靠最后一步。
4. 8K 下游 benchmark（基准评测）是否超过 1-mer composition（单碱基组成）、CNN（卷积神经网络）、公开模型和 no-region（无区域辅助）消融。
5. TE/repeat（转座元件/重复序列）相关任务是否有可靠 EDTA（转座元件注释软件）证据支撑。

评估: 预训练 loss（损失）下降只是必要条件，不是充分条件；正式结论必须来自冻结表示或微调后的独立下游任务，并且需要基线和消融共同支持。
