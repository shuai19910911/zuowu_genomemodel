# CropGenome-FM：作物基因组基础模型

**状态：Stage B 8K 已冻结，GFF-derived CropGenome-Bench v1 正式评估完成，Stage C1 64K gate 通过。**

本项目面向多作物基因组序列预训练，目标是学习可迁移到启动子、剪接、转录终止、基因结构、变异效应和育种任务的通用表示。当前阶段的主要创新定位是“作物基因组预训练模型”；下游任务用于检验作物预训练相对通用 DNA 模型的实际价值。

## 快速入口

- [训练进展与当前决策](TRAINING_PROGRESS.md)
- [CropGenome-Bench v1 正式结果：小白版详细解读](docs/training_progress/cropgenome_bench_v1_formal_a100/README.md)
- [正式结果全量对比图](docs/training_progress/cropgenome_bench_v1_formal_a100/figures/formal_full_data_balanced_accuracy.png)
- [正式少样本对比图](docs/training_progress/cropgenome_bench_v1_formal_a100/figures/formal_fewshot_balanced_accuracy.png)
- [完整项目计划](PROJECT_PLAN.md)
- [模型结构说明](MODEL_ARCHITECTURE.md)

## 当前正式结果

本次 CropGenome-Bench v1 使用原始 GFF/GTF 注释坐标和 FASTA 构建 3 个 hard-negative（硬负样本）二分类任务。每任务 6,144 条 512 bp 序列，train/validation/test=`4096/1024/1024`，正负平衡；训练、验证和测试物种严格不重叠。所有模型统一使用 frozen embedding + linear probe（冻结向量 + 线性分类头）。

### 100% 标签 balanced accuracy（平衡准确率）

| 任务 | Best k-mer | DNABERT-2 | NT-v2 100M* | step14000 | step17000 |
|---|---:|---:|---:|---:|---:|
| promoter/TSS | 0.6113 | 0.6494 | 0.6689 | 0.6875 | **0.6885** |
| splice donor/acceptor | 0.6797 | 0.7090 | 0.7158 | **0.8896** | 0.8672 |
| TES/poly(A) | 0.6289 | 0.6182 | **0.6592** | 0.6387 | 0.6455 |
| 三任务平均 | 0.6400 | 0.6589 | 0.6813 | **0.7386** | 0.7337 |

`*` NT-v2 是读取预锁定主结果后追加的同口径补充模型，标记为 post-hoc supplementary（事后补充），不用于反向选择正式 checkpoint。

核心结论：

- splice 是当前最强证据：step14000 相对 DNABERT-2 提升 `18.07` 个百分点，相对 NT-v2 提升 `17.38` 个百分点。
- promoter 有中等提升：step17000 相对 DNABERT-2 提升 `3.91` 个百分点，相对 NT-v2 提升 `1.95` 个百分点。
- TES 是当前短板：step17000 高于 DNABERT-2 和最佳 k-mer，但低于 NT-v2 `1.37` 个百分点。
- step14000/17000 各有优势，不存在单一 checkpoint 全任务占优。
- 1% 标签时，step17000 三任务平均为 `0.6669`，最强公开模型为 `0.5663`；少样本优势比全量标签更明显。

![CropGenome-Bench v1 正式结果](docs/training_progress/cropgenome_bench_v1_formal_a100/figures/formal_full_data_balanced_accuracy.png)

详细指标含义、逐任务白话解读、少样本 mean±SD（平均值±标准差）、RC 稳健性和限制见 [正式结果详细报告](docs/training_progress/cropgenome_bench_v1_formal_a100/README.md)。

## Stage C1 64K gate

A100 GPU2 已真实完成 `[1, 65536]` 输入的 forward/backward/optimizer step（前向、反向和参数更新）：

- step14000 validation-best checkpoint 的 430 个参数键全部匹配；
- loss=`0.631388`，数值有限；
- 峰值显存 allocated/reserved=`29,240.5/30,762.0 MiB`；
- 工程 gate：**PASS**；下游有效性 gate：**PASS**；Stage C1 正式训练：**GO**。

这只证明 64K 能运行，不证明长上下文已经更准。64K 相对 8K 的实际收益需要 Stage C1 独立验证和长程任务证明。

## 研究路线

### 多作物统一预训练

主干输入覆盖作物核基因组、叶绿体基因组和线粒体基因组，同时保留来源标记和质量审计，避免把多来源数据简单混合后失去可追溯性。

### 分阶段上下文扩展

- Stage A：2K context，稳定学习局部序列规律；
- Stage B：8K context，学习更长启动子、基因结构和转座元件上下文；
- Stage C1：64K context，面向长内含子、完整基因结构和远距离调控；
- Stage C2：128K context，在有明确长程收益后再扩展。

### 下游评价

评价不只看 MLM loss（掩码语言模型损失），还包括：

1. promoter/TSS、splice、TES/poly(A) 等功能元件任务；
2. 1%/10%/100% 标签效率；
3. 物种不相交迁移；
4. reverse-complement robustness（反向互补稳健性）；
5. k-mer、random-init 和公开 DNA 模型基线；
6. 后续长上下文、TE boundary、变异效应和育种任务。

## 当前模型

`CropGenome-FM-v2-Stable-8K` 的核心设计：

- 约 220M 参数；
- `d_model=768`，`num_layers=14`，`num_heads=12`；
- GQA（分组查询注意力）；
- RoPE（旋转位置编码）；
- 8K 稀疏注意力；
- species/region/quality embeddings（物种、区域和质量嵌入）；
- RC-equivariant design（反向互补等变设计）；
- bf16 mixed precision（bf16 混合精度）；
- activation checkpointing（激活重计算）；
- MLM 为主目标，RC 为辅助约束。

## 当前决策

1. Stage B 在 step17000 停止，不继续盲目加训。
2. 论文正式结果保留 step14000 和 step17000，不隐去任务级差异。
3. Stage C1 初始化使用 validation-best `checkpoint_best.pt = step14000`，避免用 formal test 选初始化点。
4. 论文不能写“所有任务都优于公开模型”，因为 TES/poly(A) 低于 NT-v2。
5. 后续优先证明 64K 长上下文的真实收益，而不是继续反复触碰现有 formal test。

## 仓库发布边界

GitHub 仅保留轻量、可复核材料：Markdown、聚合 TSV/JSON 和 PNG。原始 FASTA/GFF、embedding cache、逐样本预测、checkpoint 和训练日志留在训练存储，不提交到仓库。
