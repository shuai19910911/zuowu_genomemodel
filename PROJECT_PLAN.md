# CropGenome-FM 研究详细方案

更新时间: 2026-06-29 08:20 CST

![CropGenome-FM 研究方案图](assets/cropgenome_fm_roadmap.svg)

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
| Stage B | 8K | 局部基因结构、剪接、启动子、UTR、TES、背景区域学习 | v2 Stable 已到 step2530，step2000 为当前 best checkpoint（最佳模型存档点） |
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

## 6. 下游 benchmark（基准评测）设计

第一波任务:

| 任务 | 目标 | 主要指标 | 解释与评估 |
|---|---|---|---|
| splice donor/acceptor（剪接供体/受体） | 判断剪接边界 | AUROC（排序区分能力）、AUPRC（阳性检出能力）、F1（精确率和召回率综合） | 作物基因结构核心任务；若这里无提升，模型生物语法价值不足。 |
| promoter/TSS（启动子/转录起始位点） | 识别启动子和起始区域 | AUROC/AUPRC/F1 | 受上下文和物种差异影响，适合检验 crop-specific pretraining（作物专用预训练）。 |
| TES/polyA（转录终止/多聚腺苷酸化） | 识别转录终止相关区域 | AUROC/AUPRC/F1 | 终止信号通常弱于剪接，能检验模型是否学到非编码调控模式。 |
| exon/intron/UTR boundary（外显子/内含子/非翻译区边界） | 单碱基或区域注释 | Macro-F1（类别平均 F1）、balanced accuracy（类别平均召回） | 类别不平衡明显，不能只看 accuracy（准确率）。 |
| TIS/TTS（翻译起始/终止位点） | 识别编码起止边界 | AUROC/AUPRC/F1 | 直接检验 CDS（编码区）语法学习。 |
| TE insertion boundary（转座元件插入边界） | 识别转座元件边界 | AUROC/AUPRC/F1 | 必须等 EDTA（转座元件注释软件）证据可靠后执行，否则不进入主结论。 |
| cross-species transfer（跨物种迁移） | 留出物种/属泛化 | 每任务主指标 + species/genus 分层 | 论文最重要的泛化证据。 |

解释: downstream benchmark（下游基准评测）必须是独立评测，不是预训练时的 region bucket（区域桶）标签自测。每个任务都要有单独曲线、源数据和解释，但对外展示时全部汇总进 [TRAINING_PROGRESS.md](TRAINING_PROGRESS.md)。

评估: 小样本 probe（探针评测）只用于早期方向判断；正式 benchmark（基准评测）必须固定 split（数据划分）、固定指标、固定 baseline（基线）、固定 checkpoint（模型存档点）选择规则。

## 7. 基线和消融

必须包含:

- 1-mer composition（单碱基组成）: 最低限度基线。
- CNN（卷积神经网络）监督基线: 检查简单监督模型能否已解决任务。
- DNABERT-2 / AgroNT / PlantCAD2 / HyenaDNA / Evo2（公开 DNA 或植物基因组模型）: 按可获得性和资源执行。
- v2 no-region（无区域辅助）: 移除 region auxiliary head（区域辅助头）。
- v2 with-region（有区域辅助）: 当前主模型。
- 旧 formal CaduceusRC（反向互补一致性旧模型）: 只作历史附录，不作为 v2 输入。

解释: 如果 v2 只超过 1-mer（单碱基组成），但不超过 CNN 或公开模型，论文说服力不足。如果 with-region 明显超过 no-region，也必须证明不是标签泄漏或任务同源导致虚高。

评估: 消融是防 shortcut（捷径学习）的核心。正式表格应把外部公开模型作为主比较，把内部版本差异放入消融表，避免主文混乱。

## 8. 成功标准

最低成功标准:

1. Stage B v2 Stable 能稳定训练到至少 step1000 并产生 validation（验证）和 checkpoint（模型存档点）。
2. selection loss（选择损失）在早期呈下降趋势。
3. 至少 2-3 个独立下游任务超过 1-mer（单碱基组成）和 CNN（卷积神经网络）基线。
4. no-region/with-region（无/有区域辅助）消融不显示明显标签泄漏。
5. 跨物种或跨属 split（划分）中仍保留收益。

理想成功标准:

1. 在 splice、promoter/TSS、TES、exon/intron/UTR 多任务上稳定优于公开 DNA/植物模型。
2. 8K 完整上下文任务优于 128 bp（碱基对）短探针，说明长上下文有价值。
3. TE/repeat（转座元件/重复序列）任务在可靠 EDTA 注释后提供结构基因组特色结果。
4. 64K/128K 继续训练带来跨基因或长程调控任务收益。

解释: 预训练 loss（损失）下降是必要条件，但不是成功标准。最终成功必须由下游任务、消融和跨物种泛化共同决定。

评估: 若 v2 只在 region bucket（区域桶）probe 上有效，不能写成正式模型成功；若独立 benchmark 成功，即使架构不新，也可以形成作物专用基础模型论文。

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
