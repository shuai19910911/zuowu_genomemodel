# CropGenome-FM 研究详细方案

更新时间: 2026-06-29 10:45 CST

## 0. 一句话定位

CropGenome-FM（Crop Genome Foundation Model，作物基因组基础模型）要解决的问题是：能否用结构注释完整的作物基因组，训练一个长上下文 DNA language model（DNA 语言模型），让它在作物剪接、启动子、终止、基因结构、转座元件边界和跨物种迁移任务中，比简单序列组成、CNN（卷积神经网络）和公开通用 DNA 模型更有用。

解释: 项目重点不是“模型结构看起来多新”，而是“作物专用数据 + 严格防泄漏 + 独立下游 benchmark（基准评测）”。

评估: 这个定位更稳。架构创新很难单独说服审稿人，但如果能证明作物专用预训练在多个跨物种任务上稳定超过基线，就更容易形成高质量论文故事。

## 1. 数据口径

| 项目 | 当前口径 |
|---|---:|
| 有 genome FASTA（基因组序列）且有 GFF3/GTF（结构注释）的 crop assembly manifest（作物组装清单） | 263 行 |
| canonical assembly accession（标准化组装版本号） | 258 个 |
| 覆盖属 | 26 个 |
| accession 级 train/val/test split（训练/验证/测试划分） | 192 / 35 / 31 |
| Stage B（8192 碱基上下文）token（训练 token 数） | 41.24B |
| 训练服务器 transfer（搬运）目录 | 约 67G，GitHub 不上传 |

解释: 第一版正式训练只使用结构注释完整的作物基因组。没有 GFF3/GTF 的 genome（基因组）暂时放弃，因为无法可靠构建 CDS（编码序列）、splice（剪接）、UTR（非翻译区）、promoter（启动子）、TES（转录终止区）等功能区域任务。

评估: 这样会损失数据量，但能显著提高标签可靠性和下游可解释性。项目不应宣传“最大规模植物基因组预训练”，而应宣传“结构注释感知、作物专用、防泄漏的预训练与评测框架”。

## 2. 数据质量控制和区域构建

### 2.1 FASTA（基因组序列）质量控制

保留原则:

- contig/chromosome（染色体/序列片段）长度太短的区域不进入训练。
- N fraction（未知碱基比例）过高的窗口丢弃。
- 连续 N（连续未知碱基）过长的窗口丢弃。
- 极端 GC（鸟嘌呤/胞嘧啶比例）和低复杂度窗口降权或丢弃。
- organelle（细胞器）、plastid（质体）、mitochondrial（线粒体）候选序列不混入核基因组主训练。

解释: DNA language model（DNA 语言模型）很容易学到测序/组装噪声。如果不做质量控制，模型会把 N、低复杂度片段或细胞器序列当成作物核基因组规律。

评估: 严格 QC（质量控制）会减少 token（训练片段）数量，但提升训练信号质量。对第一版模型来说，质量优先于覆盖率。

### 2.2 GFF/GTF（结构注释）解析

构建区域:

- CDS/exon（编码区/外显子）
- splice donor/acceptor（剪接供体/受体）及上下游 flanking sequence（侧翼序列）
- UTR（非翻译区）
- promoter/TSS（启动子/转录起始位点）
- TES/polyA（转录终止区/多聚腺苷酸化位点）
- intron（内含子）
- high-quality intergenic（高质量基因间区）
- background（背景区域）
- TE/repeat（转座元件/重复序列），仅在 EDTA（转座元件注释软件）或可靠注释完成后启用

解释: 这些区域决定模型在预训练时看到什么样的功能信号。尤其是 splice、promoter、TES 和 UTR，直接对应后续下游任务。

评估: GFF/GTF 注释质量不均一是主要风险。区域辅助标签只能作为 weak supervision（弱监督），不能直接当作正式评测结论；正式结论必须来自独立 benchmark（基准评测）。

## 3. 防泄漏 split（数据划分）

当前 split（划分）原则:

1. 先按 canonical assembly accession（标准化组装版本）划分 train/val/test。
2. 再在各 split 内做窗口化和采样。
3. 同一 assembly accession 不允许跨 train/val/test。
4. 后续跨物种任务还要增加 held-out species/genus（留出物种/属）评测。

解释: 作物基因组存在大量近重复、同源基因和保守区域。如果先切窗口再随机划分，很容易让几乎相同的序列同时出现在 train（训练集）和 test（测试集），造成虚高结果。

评估: accession 级 split 是最低要求；论文正式 benchmark（基准评测）还需要 species/genus holdout（物种/属留出）和必要的近重复检查。只有这样，下游结果才能解释为泛化能力，而不是记忆能力。

## 4. 预训练阶段

| Stage（阶段） | context（上下文长度） | 用途 | 当前状态 |
|---|---:|---|---|
| Stage B | 8K | 局部基因结构、剪接、启动子、UTR、TES、背景区域学习 | v2 Stable 已到 step2710，step2000 为当前 best checkpoint（最佳模型存档点） |
| Stage C1 | 64K | 更长 gene body（基因体）和局部调控上下文 | 待 Stage B 稳定后启动 |
| Stage C2 | 128K | 更长距离调控、跨基因区域和结构区域 | 待 Stage C1 后启动 |
| Stage D | 256K | 超长上下文探索 | 资源允许时执行 |

解释: 8K 是稳健起点，可以先验证训练脚本、loss（损失函数）、checkpoint（模型存档点）和下游任务是否有效。64K/128K 才更接近长程调控和结构基因组问题，但成本更高。

评估: 不应一开始就追求 128K/256K。若 8K 下游 benchmark（基准评测）都没有稳定收益，扩长上下文只会增加成本。Stage B 的成功标准是 validation（验证）稳定、best checkpoint（最佳模型存档点）出现、下游 probe（探针评测）优于基础基线。

## 5. 当前主模型训练策略

当前版本: `CropGenome-FM-v2-Stable-8K`（第二版稳健 8192 碱基版）。

训练规则:

- 正式 v2 Stable 从头训练，不使用旧 `formal_caduceus_rc` checkpoint（模型存档点）warm-start（热启动）。
- `resume`（继续训练）只允许用于同一个 v2 run（第二版训练）意外中断后的严格恢复。
- 主选择指标是 `selection_loss = MLM loss + 0.02 × RC loss`。
- `region_loss`（区域辅助损失）只监控，不参与 best checkpoint（最佳模型存档点）选择。
- early stopping（早停）启用: `min_steps=5000`、`patience_evals=3`、`min_delta=0.002`。

解释: 旧模型已经有一些结果，但直接 warm-start 会让 v2 结论混入旧架构、旧训练策略和旧错误。v2 从头训练可以保证结论清晰。

评估: 从头训练成本更高，但论文结论更干净。selection loss（选择损失）不包含 region loss（区域辅助损失）是关键设计，因为 region bucket（区域桶）标签来自预训练元数据，不能让它决定“最佳模型”。

## 6. CropGenome-Bench v1 正式下游评估体系

正式任务注册文件: [`training_server_transfer/configs/downstream_v2_benchmark.json`](training_server_transfer/configs/downstream_v2_benchmark.json)。

核心问题不是“我们的模型能不能跑出一个 probe（探针）分数”，而是:

> 在同一作物任务、同一 split（划分）、同一 metric（指标）、同一下游头和同一调参预算下，CropGenome-FM 是否比通用 DNA 大模型更适合作物基因组？

### 6.1 为什么当前 step1000 probe 不够

当前 full-region annotation probe（完整区域注释探针）只用于 health check（健康检查）:

- 样本小，任务是我们内部构造的 7 类区域识别。
- 它能检查 embedding（向量表示）是否有早期信号，但不能证明作物任务优势。
- 它不能直接与 HyenaDNA、DNABERT-2、Caduceus、Nucleotide Transformer 等论文中的 GenomicBenchmarks/GUE/NT tasks 数字比较，因为任务、数据、split 和 metric 都不同。
- region head（区域预测头）来自预训练辅助标签，只能作训练接线检查，不能作为论文主结果。

评估口径: step1000 probe 是“弱阳性诊断”，正式论文结论必须来自下面的固定 benchmark（基准评测）和统一 baseline（基线）。

### 6.2 任务组 A: 作物核心基因结构语法

| 任务 | 目标 | 关键设计 | 主指标 | 为什么能证明模型价值 |
|---|---|---|---|---|
| splice donor/acceptor hard negative（剪接供体/受体硬负样本） | 判断真实剪接边界 | 负样本也含 GT/AG motif（序列基序），并匹配 GC/长度/物种 | AUPRC、MCC、AUROC | 防止模型只看 GT/AG 两个碱基取巧。 |
| exon/intron/UTR segmentation（外显子/内含子/UTR 序列标注） | 按位置标注基因结构 | token-level（逐碱基/逐 token）或 segment-level（片段级）标注 | Macro-F1、boundary-F1、segment-IoU | 直接检验是否学到作物基因结构边界。 |
| TIS/TTS context（翻译起始/终止上下文） | 识别 CDS 起止位点 | 用同框/同密码子 decoy（诱饵负样本） | AUPRC、MCC、F1 | 防止只记 ATG/TAA/TAG/TGA，要求理解编码区上下文。 |

### 6.3 任务组 B: 作物调控元件

| 任务 | 目标 | 关键设计 | 主指标 | 为什么适合作物预训练 |
|---|---|---|---|---|
| promoter/TSS hard negative（启动子/转录起始位点） | 识别启动子 | 负样本匹配 GC/长度，并加入 shifted promoter decoy（偏移启动子诱饵） | AUPRC、AUROC、calibration-ECE | 作物启动子和人类/细菌数据不同，是作物域优势核心。 |
| TES/polyA context（转录终止/多聚腺苷酸化） | 识别转录终止信号 | 加入 AT-rich non-boundary（富 AT 但非边界）负样本 | AUPRC、MCC、AUROC | 终止信号弱，能检验非编码上下文学习。 |
| ATAC/ACR open chromatin（开放染色质峰） | 预测调控开放区域 | 只用 processed ATAC/ACR（处理后峰文件），先做 assembly compatibility QC（组装兼容质控） | AUPRC、AUROC、MCC | 若低样本/跨作物仍好，说明作物调控预训练有效。 |
| tissue/stress responsive promoter（组织/逆境响应启动子） | 根据启动子序列预测表达响应类别 | 表达标签必须 fold-aware（只用训练折统计） | Macro-F1、AUPRC、Spearman | 连接序列到作物表达调控，是更强生物功能证据。 |

### 6.4 任务组 C: 作物结构基因组特色

| 任务 | 目标 | 执行前提 | 主指标 | 解释 |
|---|---|---|---|---|
| TE boundary high-confidence EDTA（高置信转座元件边界） | 识别 TE 插入边界 | EDTA（转座元件注释软件）通过 QC 后才进入主结论 | boundary-F1、AUPRC、family-stratified F1 | 作物基因组 TE 占比高，这是通用人类 DNA 模型未必擅长的任务。 |
| gene boundary long context（长上下文基因边界） | 用 8K/64K/128K 上下文识别基因起止 | Stage C/D 产生后比较不同输入长度 | boundary-F1、segment-IoU | 用来证明长上下文不是摆设。 |

### 6.5 任务组 D: 变异效应和育种相关排序

| 任务 | 目标 | 数据原则 | 主指标 | 解释 |
|---|---|---|---|---|
| variant effect ref/alt delta（SNP/indel 变异效应排序） | 比较 ref/alt 等位序列的模型分数变化 | 只用 published processed VCF/GWAS/QTL/eQTL 表，不下载 WGS/FASTQ/BAM 原始重测序 | NDCG@10、top-k recall、enrichment-fold、Spearman | 检验模型是否能把功能变异排到前面。 |
| QTL/GWAS candidate gene ranking（候选基因排序） | 在 QTL/GWAS 区间内排序候选基因 | 用已发表候选基因/区间，不重做变异 calling | precision@k、NDCG@10、enrichment-fold | 更贴近育种应用，也更能体现作物模型定位。 |

### 6.6 任务组 E: 跨作物迁移和低样本效率

| 协议 | 做法 | 主指标 | 论文价值 |
|---|---|---|---|
| leave-one-crop/species out（留一作物/物种外推） | 一个物种/属完全不参与训练，只用于测试 | cross-species score、retention ratio（跨物种保留率） | 这是证明“作物预训练”泛化的主证据。 |
| monocot-to-dicot / dicot-to-monocot（单子叶到双子叶/反向迁移） | 按植物大类划分训练和测试 | AUPRC/Macro-F1 保留率 | 检验模型是否学到跨作物规律。 |
| few-label efficiency（少标签效率） | 只用 1%/5%/10%/50% 标注训练下游头 | few-shot gain（少样本增益） | 基础模型理论上应在标注少时更有优势。 |

## 7. 与其他模型的公平比较方案

### 7.1 对比模型

| 类别 | 模型/方法 | 用途 | 注意事项 |
|---|---|---|---|
| 简单基线 | random（随机）、majority（多数类）、1-mer/2-mer/3-mer composition（碱基组成） | 给所有数字一个最低参照 | 必须每个任务都有。 |
| 传统监督模型 | k-mer logistic/SVM、small CNN（小卷积网络） | 判断任务是否已被简单模型解决 | 若 CNN 已接近上限，不能夸大基础模型贡献。 |
| 通用 DNA 大模型 | DNABERT-2、Nucleotide Transformer、HyenaDNA、Caduceus、Evo2（若资源可行） | 主外部比较 | 必须同任务、同 split、同下游头；不能拿论文旧表格直接比胜负。 |
| 植物/作物 DNA 模型 | AgroNT、PlantCaduceus/PlantCAD 等可获得权重 | 最关键外部比较 | 若权重不可用，记录为 not runnable（不可运行），不能伪造。 |
| 我们的模型 | CropGenome-FM frozen embedding（冻结向量）和 full fine-tuning（全量微调） | 检验预训练质量和微调上限 | 主表优先 frozen/probe，fine-tuning 作为补充。 |
| 我们的消融 | no-pretrain、no-region、no-RC、短上下文 512bp/1kb/2kb、单作物预训练 vs 多作物预训练 | 解释为什么有效 | 内部版本放消融表，不混进主外部比较表。 |

### 7.2 两层比较，避免不公平

1. Frozen encoder comparison（冻结编码器比较）: 所有模型只提取 embedding（向量），接同一个 logistic/MLP/head（下游头）。这是最公平的“预训练表示质量”比较。
2. Full fine-tuning comparison（全量微调比较）: 所有模型允许微调，但必须同 epoch（轮数）、同 batch（批量）、同 early stopping（早停）、同随机种子。这个比较成本高，作为强证据或补充。

严禁做法:

- 不拿我们的自定义 probe 和别人论文的 GenomicBenchmarks/GUE 数字直接比胜负。
- 不用 test（测试集）选择 checkpoint（模型存档点）或调参。
- 不让某个模型用更长输入、更大调参预算或更多标注数据。
- 不把 region head（区域辅助头）结果当正式下游成功。

### 7.3 split（划分）和防泄漏

正式 benchmark 至少包含四种划分:

| split | 中文解释 | 解决什么问题 |
|---|---|---|
| assembly holdout | 留出组装版本 | 防止同一组装窗口泄漏。 |
| gene-family / orthogroup holdout | 留出基因家族/直系同源组 | 防止同源基因记忆。 |
| species/genus holdout | 留出物种/属 | 证明跨作物泛化。 |
| low-homology holdout | 留出低同源序列 | 检查模型是否只靠相似序列检索。 |

所有 normalization（归一化）、negative sampling（负样本采样）、feature selection（特征筛选）都必须只用 train split（训练集）拟合。

## 8. 成功标准和论文主表规则

### 8.1 最低成功标准

1. Stage B v2 Stable 稳定训练到 step5000，并有 step3000/5000 固定 benchmark 结果。
2. 至少 3 个 P0 作物任务完成固定 test（测试集）评估。
3. CropGenome-FM 在同口径任务上平均超过最强可运行通用 DNA/植物 DNA 模型至少 3% relative improvement（相对提升）。
4. 至少一个 species/genus holdout 或 low-homology holdout 中仍有收益。
5. 5 个随机种子报告 mean ± std（均值±标准差），不能只报单 seed（随机种子）。
6. random/majority/k-mer/CNN/no-pretrain/no-region 基线齐全。
7. full-region probe 明确标注为 diagnostic only（仅诊断），不进入论文主表。

### 8.2 强论文标准

1. 在 splice、promoter/TSS、TES、exon/intron/UTR 中至少 3 类任务稳定优于公开模型。
2. 在 few-shot（少标签）或 cross-crop transfer（跨作物迁移）中优势更大，符合“作物预训练模型”的理论预期。
3. TE boundary 或 QTL/GWAS candidate ranking 至少一个作物特色任务提供额外亮点。
4. 8K/64K/128K 长上下文对 gene boundary、TE boundary 或变异排序有清晰收益。
5. 消融证明 crop pretraining（作物预训练）> no-pretrain（无预训练）> 简单 k-mer/CNN，且 no-region 不会暴露标签泄漏。

### 8.3 主表和补充表

| 表格 | 放什么 | 不放什么 |
|---|---|---|
| 主表 1 | P0 作物任务同口径外部模型比较 | 内部旧版本、失败实验、不同任务论文数字 |
| 主表 2 | 跨作物/少标签迁移结果 | 只在随机 split 好看的结果 |
| 主表 3 | 作物特色任务: TE/variant/QTL 排序 | 标签质量未过 QC 的任务 |
| 消融表 | no-pretrain/no-region/no-RC/短上下文/单作物 vs 多作物 | 外部模型主比较 |
| 补充表 | per-species、per-class、confusion matrix、失败/不可运行模型说明 | 不上传 GitHub 明细目录，只在本地或补充材料整理 |

解释: 成功不是“训练 loss 下降”，而是“在作物任务上，用公平协议稳定超过可运行的外部 DNA 模型和强基线”。

## 9. 风险和应对

| 风险 | 表现 | 应对 | 评估 |
|---|---|---|---|
| 结构注释噪声 | 不同物种 GFF/GTF 质量差异大 | 坐标合法性、biotype（生物类型）过滤、低质量区域降权 | 必须记录过滤比例，避免把注释错误当生物信号。 |
| 数据泄漏 | 相似序列跨 split | accession/species/genus 级 split，必要时近重复去重 | 下游结果必须说明 split 口径。 |
| region head 虚高 | 区域辅助任务与下游任务过近 | region loss 不参与 best checkpoint，必须做 no-region 消融 | region 只能当辅助，不是主创新。 |
| 训练资源不足 | 8K 尚未稳定就扩长 | 先完成 Stage B 和第一波 benchmark | 不为追求长上下文牺牲验证质量。 |
| TE 注释不完整 | TE boundary 任务标签不可靠 | EDTA 只用可靠完成目标；失败大基因组延后 | TE 相关结果可作为补充，不强行进入主结论。 |
| 公开模型基线难跑 | 模型大、环境复杂 | 先跑可获得模型和轻量 baseline，再补强 | 不能因基线难跑就省略主比较。 |

## 10. 对外展示原则

- README 只讲项目定位和入口。
- PROJECT_PLAN 讲完整研究方案。
- MODEL_ARCHITECTURE 讲模型结构和边界。
- TRAINING_PROGRESS 是唯一进展入口，所有训练图表、下游 probe（探针评测）摘要、结构注释进度和评估都写进去；逐 checkpoint 明细目录只本地保留。
- GitHub 不上传 raw data（原始数据）、checkpoint（模型存档点）、大日志、训练输入 shard（分片）或逐样本预测。

解释: 目录越少，读者越容易抓住主线。分散的旧 probe 目录、临时 EDTA 记录和迁移说明会保留在本地历史中，但不再作为 GitHub 入口。

评估: 这种结构更像论文项目主页：项目介绍、方案、架构、进展四件事清楚分开，所有结果集中在一个进展文档中，便于导师、合作者或审稿前自查。
