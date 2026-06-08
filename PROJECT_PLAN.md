# 作物基因组预训练大模型详细训练方案

更新时间: 2026-06-08 11:42:31 CST

## 1. 当前决策

本项目第一版正式模型只使用有结构注释的基因组。此前 1906 个 genome 中，1644 个缺少 GFF3/GTF 注释的 assembly 暂时放弃，不进入预训练。这样牺牲纯序列规模，换来更高质量的区域定义、区域加权预训练、严格下游任务构建和更可控的数据泄漏风险。

当前正式数据口径:

- 使用: 262 个同时具备 genome FASTA 和 GFF3/GTF 的 assembly。
- 覆盖: 26 个属。
- assembly level: chromosome 233 个、complete genome 13 个、scaffold 12 个、contig 4 个。
- 原始压缩体积: genome `.fna.gz` 约 213.48 GB，GFF3/GTF `.gz` 约 5.40 GB，合计约 218.89 GB。
- 不使用: 1644 个只有 genome、缺少结构注释的 assembly。

训练目标不是复刻短窗口 DNA BERT，而是训练一个结构注释感知、区域加权、长上下文、反向互补一致的作物基因组基础模型。

## 2. 文献和前沿方向

当前模型设计依据:

- AgroNT: 植物 DNA LM 强基线，48 个植物 genome，约 6000 bp 输入、6-mer token、15% MLM，在植物调控和变异优先级任务上有效。来源: https://www.nature.com/articles/s42003-024-06465-2
- PlantCAD/PlantCAD2: 植物 Caduceus/Mamba 路线，强调单碱基、双向、反向互补等变。PlantCAD2 提供 Mamba2 植物长上下文模型配置。来源: https://github.com/plantcad/plantcad
- Evo 2 / StripedHyena 2: 2026 年大规模基因组基础模型代表，强调单碱基、长上下文、短到长 midtraining 和 likelihood 变异评分。来源: https://www.nature.com/articles/s41586-026-10176-5
- HyenaDNA: 证明单碱基长上下文 DNA 建模可行，支持 1M token 级别上下文。来源: https://arxiv.org/abs/2306.15794
- HybriDNA: Transformer + Mamba2 hybrid 思路，适合同时保留注意力的局部精确交互和 SSM 的长程效率。来源: https://arxiv.org/abs/2502.10807
- AlphaGenome: 监督式 1Mb DNA 输入多模态调控预测模型，说明长上下文对调控、表达和变异任务的重要性。来源: https://www.alphagenome.com/
- DNABERT-2/GROVER: BPE/k-mer DNA tokenization 是重要基线，但单碱基建模更适合 ref/alt 变异解释。来源: https://arxiv.org/abs/2306.15006 和 https://www.nature.com/articles/s42256-024-00872-0

结论: 主模型采用 `RC-equivariant Mamba2/Hyena + periodic attention + region-weighted MLM/causal objectives`。AgroNT、DNABERT-2、PlantCAD2、HyenaDNA 和 CNN/DeepSEA-like 模型作为基线。

## 3. 数据预处理总流程

所有新生成脚本、manifest、索引、shard、日志和结果都放在当前项目目录下。原始 plantDB 目录只读。

### 3.1 产物目录

| 目录 | 内容 | 预计体积 | GitHub |
|---|---|---:|---|
| `data_manifests/` | 262 个 assembly manifest、split、区域权重表 | < 1 GB | 不上传 |
| `sequence_index/` | contig 长度、GC、N、softmask、offset | 1-5 GB | 不上传 |
| `annotation_index/` | gene/transcript/exon/CDS/UTR/intron/intergenic/promoter/TSS/TES 区间 | 10-80 GB | 不上传 |
| `window_index/` | 训练窗口坐标、区域类型、质量指标、split、权重 | 20-100 GB | 不上传 |
| `token_shards/` | uint8 token shards、labels/mask 可动态生成 | 0.6-2.5 TB | 不上传 |
| `configs/` | 关键训练配置 | < 100 MB | 可选择上传摘要 |
| `logs/` | Slurm 和训练日志 | 10-200 GB | 不上传 |
| `checkpoints/` | 模型和 optimizer checkpoint | 0.5-5 TB | 不上传 |
| `results/` | 下游评测和表格 | 10-500 GB | 只上传总结 |

## 4. 预处理步骤、资源和时间

| 步骤 | 输入 | 输出 | 过滤/处理 | 资源 | 预计时间 |
|---|---|---|---|---|---|
| A1 manifest | completed index | `assemblies.tsv` | 只保留有 genome + GFF/GTF 的 262 个 | 8 核 32G | 5-20 分钟 |
| A2 FASTA 扫描 | 213GB `.fna.gz` | `contigs.tsv` | 统计长度、N、GC、softmask、header | 6 个 CPU 作业，各 30 核 100-150G | 8-24 小时 |
| A3 注释解析 | 5.4GB GFF/GTF | gene/transcript/CDS/UTR/intron 表 | 校验层级、坐标、strand、CDS phase | 2-6 个 CPU 作业，各 30 核 80-150G | 8-36 小时 |
| A4 区域构建 | FASTA index + 注释 | promoter/TSS/TES/intergenic/region map | 合并重叠、去除冲突、多转录本取 canonical + 保留 isoform 标签 | 2-6 个 CPU 作业，各 30 核 80-150G | 6-24 小时 |
| A5 split | assembly + gene + region | split 表 | assembly/species/genus holdout；窗口不跨 split | 8 核 32G | 0.5-2 小时 |
| A6 窗口候选 | region map | `window_index/*.parquet` | 按区域和质量采样，不全量输入 | 6 个 CPU 作业，各 30 核 150G | 8-36 小时 |
| A7 token shard | window index + FASTA | `token_shards/` | A/C/G/T/N -> uint8；保存 metadata | 6 个 CPU 作业，各 30 核 150G | 12-48 小时 |
| A8 训练 dry-run | token shards | 1000 step 吞吐报告 | 验证 batch、mask、loss、split | 1-2 GPU | 2-8 小时 |

CPU 预处理总耗时预计 2-5 天。若共享文件系统吞吐低，最慢步骤会是 A2 和 A7。

## 5. 片段过滤标准

不会把所有序列都输入模型。候选窗口必须先通过 assembly、contig、annotation 和 window 四层过滤。

### 5.1 Assembly 过滤

- 必须有 genome FASTA 和 GFF3/GTF。
- 优先 chromosome/complete genome；scaffold 和 contig assembly 只保留注释一致性高、N 比例低的区域。
- 同一物种/品种存在多个版本时，优先 RefSeq、chromosome-level、新版本；旧版本只作为 holdout 或低权重。

### 5.2 Contig 过滤

- 长度 `< 10 kb` 的 contig 不进入预训练。
- contig N fraction `> 10%` 不进入预训练；`5%-10%` 只允许非关键区域低权重采样。
- 极端 GC: 属内 GC z-score `>|4|` 的 contig 进入人工检查或低权重。
- organelle/plastid/mitochondrial contig 单独标记，第一版主模型不混入核基因组训练。

### 5.3 Annotation 过滤

- gene/transcript/exon/CDS 坐标必须合法且 strand 一致。
- CDS phase 严重冲突、exon 顺序错误、转录本越界的基因不进入 CDS/TIS/TTS/splice 任务。
- 多转录本基因保留 canonical transcript 作为主训练区域；其他 isoform 只用于增强或下游 isoform 任务。
- 蛋白编码基因、lncRNA、pseudogene、transposon gene 分开标记，不混成单一类别。

### 5.4 Window 过滤

- 8K/64K/128K 窗口有效 A/C/G/T 比例必须 `>= 95%`；N 比例 `<= 5%`。
- 低复杂度窗口过滤: Shannon entropy `< 1.2` 的 1kb 子窗口占比过高则剔除。
- homopolymer 过滤: 任意单碱基连续长度 `> 100 bp` 且非真实注释区域时剔除或低权重。
- repeat/softmask 比例: CDS/TSS/splice 窗口不因 repeat 直接剔除；intergenic 窗口 softmask `> 50%` 剔除。
- 近重复控制: 与同 split 内高相似窗口重复时只保留代表；跨 split 禁止出现相似度 `>= 95%` 且 overlap `>= 80%` 的窗口。
- intergenic 只采高质量 10%: 从远离基因、低 N、低重复、GC 正常、有保守性或靠近已知调控上下文的 intergenic 候选中抽样，不做全量输入。

## 6. 严格防泄漏 split

原则: 一个数据集中的基因组片段不能同时出现在训练集和验证/测试集。

### 6.1 预训练 split

- 主 split 以 assembly 为单位: 同一 assembly 的所有窗口只能属于 train、val、test 或 holdout 中的一个。
- species holdout: 若某 species 有多个 assembly，至少保留部分 species 全量进入测试，用于跨品种/跨物种泛化。
- genus holdout: 选择小属和高价值作物属作为完整属留出，检验跨属迁移。
- split 完成后再窗口化，避免先切窗口再分组造成泄漏。

### 6.2 下游 split

- gene-level split: 同一 gene_id 的所有 promoter、CDS、UTR、splice、TIS/TTS 片段不能跨 split。
- paralog-aware split: 高相似基因家族按 family 分组，防止同源片段泄漏到测试集。
- coordinate exclusion: 若必须在同 assembly 内切分，train 与 val/test 窗口之间至少留出 `2 x max_context` 间隔。
- variant task split: 同一 variant、同一 LD block、同一 gene 周围窗口不能跨 split。

## 7. 区域加权训练设计

只使用 262 个注释完整基因组后，预训练不再是简单随机 genome window，而是结构区域感知采样。

### 7.1 Batch 区域比例

| 区域 | 批内目标比例 | loss 权重 | 说明 |
|---|---:|---:|---|
| CDS/protein-coding exon | 30% | 1.50 | 优先学习密码子、ORF、保守编码结构 |
| splice donor/acceptor 周围 | 12% | 2.00 | 预测剪接位点和外显子边界，是最重要下游任务之一 |
| UTR | 8% | 1.20 | 翻译调控和 mRNA 稳定性相关 |
| promoter/TSS upstream | 15% | 1.40 | 表达调控和启动子任务 |
| terminator/TES/polyA 周围 | 8% | 1.20 | 转录终止和 polyA 相关 |
| intron | 12% | 0.90 | 保留长程基因结构和调控上下文 |
| high-quality intergenic | 10% | 0.60 | 非编码区只采高质量 10%，避免噪声支配 |
| random genomic background | 5% | 0.50 | 保留 genome-wide 分布，防止模型只见注释区域 |

区域比例是 batch sampler 的目标比例，不代表真实 genome 比例。每个 epoch 记录实际比例并做偏差修正。

### 7.2 区域窗口构造

- CDS: 以 exon/CDS block 为中心，8K 窗口覆盖上下游；长上下文阶段拼接同一 gene body。
- splice: donor/acceptor 位点中心化窗口，短窗口用于位点，长窗口加入上下游 exon/intron。
- promoter/TSS: TSS 上游 5kb + 下游 1kb 为核心，8K/64K 窗口扩展到附近调控区。
- TES/polyA: TES 上下游各 3kb，标注转录方向。
- intron: 采样长 intron、高质量 intron、靠近 splice 的 intron 子区间；低复杂度 intron 降权。
- intergenic: 只采 N 低、重复低、GC 正常、远离 assembly gap 的候选；不全量采样。

## 8. 模型输入、架构和 loss

### 8.1 预训练输入

每个样本包含:

```text
input_ids:       [B, L] uint8/int64, A=0 C=1 G=2 T=3 N=4 MASK=5 PAD=6
labels_mlm:      [B, L] int64, 非 mask 位点为 -100
loss_mask:       [B, L] bool, N/PAD/低质量位点为 0
region_ids:      [B, L] uint8, CDS/splice/UTR/promoter/TES/intron/intergenic/background
region_weights:  [B, L] float16, 由区域和质量共同决定
rc_flag:         [B] bool, 是否输入 reverse-complement
metadata:        assembly_id, species_id, genus_id, contig_id, start, end, strand, split
```

上下文长度:

- Stage B: 8192。
- Stage C1: 65536。
- Stage C2: 131072。
- Stage D: 262144；资源充足后才尝试 512K。

### 8.2 架构

主模型 CropGenome-FM-Large:

- token embedding: 单碱基 + region embedding + optional strand/position feature。
- backbone: RC-equivariant bidirectional Mamba2/Hyena block。
- periodic attention: 每 4-6 层插入 local/global sparse attention，增强 promoter-splice-CDS 等精确相互作用。
- normalization: RMSNorm。
- MLP: gated MLP/SwiGLU。
- output heads: MLM head、causal LM auxiliary head、region-aware contrastive/probe head。
- 参数量: 300M-450M。

备选:

- Base 100M-150M: 资源不足时的正式较小模型。
- XL 800M-1.2B: 8-16 张 80GB GPU 稳定后再考虑。

### 8.3 Loss

总 loss:

```text
L_total =
  1.00 * L_region_weighted_MLM
  + 0.10 * L_causal_next_token
  + 0.05 * L_reverse_complement_consistency
  + 0.05 * L_region_contrastive_optional
```

主 loss:

- `L_region_weighted_MLM`: 对 15% mask/span 位点做 cross entropy，按 `region_weights` 加权。
- mask span: 1-512 bp；CDS/splice/TSS 可更多短 span，intron/intergenic 可混合长 span。
- N/PAD/低质量碱基不计入 loss。

辅助 loss:

- `L_causal_next_token`: 只在一部分 batch 上启用，支持 ref/alt likelihood score。
- `L_reverse_complement_consistency`: 同一窗口 forward 与 reverse-complement embedding/logits 保持一致。
- `L_region_contrastive_optional`: 同一 gene 周围不同区域做弱对比，后续视训练稳定性决定是否启用。

## 9. 训练资源和总时间估算

### 9.1 CPU 预处理

| 阶段 | 作业数 | 每作业资源 | 时间 |
|---|---:|---|---:|
| manifest + split | 1 | 8-16 核，32G | < 2 小时 |
| FASTA/annotation 扫描 | 2-6 | 30 核，80-150G | 8-36 小时 |
| 区域构建 + 窗口过滤 | 2-6 | 30 核，80-150G | 12-48 小时 |
| token shard | 2-6 | 30 核，150G | 12-48 小时 |
| 下游数据构建 | 2-6 | 30 核，80-150G | 12-72 小时 |

CPU 总体: 2-5 天。最多每批提交 6 个 CPU 命令，符合当前 q07/q08 限制。

### 9.2 GPU 训练

| 阶段 | context | token 预算 | 推荐 GPU | 预计时间 |
|---|---:|---:|---:|---:|
| Stage B | 8K | 30B-80B | 4-8 x 80GB | 5-18 天 |
| Stage C1 | 64K | 15B-40B | 8 x 80GB | 7-18 天 |
| Stage C2 | 128K | 5B-20B | 8 x 80GB | 5-14 天 |
| Stage D | 256K | 2B-10B | 8-16 x 80GB | 4-14 天 |
| 下游 LoRA/linear probe | 8K-128K | 按任务 | 1-4 x 80GB | 1-2 周 |

推荐资源下，从预处理到完成第一版正式模型和核心下游评测，预计 6-10 周。若只有 2 张 80GB GPU，预计 2-4 个月，并建议先停止在 64K/128K。

## 10. 跨服务器搬运和磁盘空间估算

若在本服务器完成数据处理，再把处理好的数据搬到其他服务器训练，建议搬运“token shard + metadata + configs”，不搬运所有中间日志和 checkpoint。

| 包 | 内容 | 估计体积 |
|---|---|---:|
| 最小训练包 | filtered token shards for 8K + metadata + configs | 0.6-1.0 TB |
| 推荐训练包 | 8K + 64K/128K token shards + annotation metadata | 1.2-2.0 TB |
| 完整训练包 | 所有 token shards、window index、annotation index、下游数据 | 2.0-3.5 TB |
| 加原始数据备份 | 完整训练包 + 218.89GB raw gzip | 2.3-3.8 TB |
| 加 checkpoint | 训练包 + 最近 3-5 个 checkpoint | 3.0-8.0 TB |

训练服务器建议可用磁盘:

- 最低: 2 TB，只能跑 8K/部分 64K。
- 推荐: 4 TB，可放推荐训练包、日志和少量 checkpoint。
- 充足: 8-12 TB，可保留完整训练包、多阶段 checkpoint 和下游 embedding。

## 11. 下游任务详细设计

| 任务 | 正例 | 负例 | split | 指标 | 预期优势 |
|---|---|---|---|---|---|
| splice donor/acceptor | GFF/GTF exon-intron junction | 同 contig 非 junction GT/AG + random | gene family + species holdout | AUROC, AUPRC, F1 | 区域加权和 splice 高 loss 权重，预计优于 DNABERT-2/AgroNT 在跨物种 donor/acceptor |
| TIS/TTS | CDS start/stop codon 周围 | 同 frame 非起止 codon | gene family holdout | AUROC, AUPRC | CDS 高权重 + 单碱基建模，预计优于 k-mer 模型 |
| promoter/TSS | TSS 上游 5kb + 下游 1kb | matched intergenic | assembly/species holdout | AUROC, AUPRC | 长上下文 + TSS 权重，预计优于短窗口 CNN 和 DNABERT-2 |
| TES/polyA | TES 周围窗口 | matched downstream negatives | gene holdout | AUROC, AUPRC | TES 专门采样，预计优于未做区域感知预训练的模型 |
| CDS/UTR/intron 区域分类 | 注释区域 | matched background | assembly holdout | macro-F1 | 区域 embedding 预训练，预计明显优于随机初始化 CNN |
| lncRNA/mRNA | transcript 注释 | 长度/GC 匹配负例 | species holdout | AUROC, MCC | 单碱基 + transcript 区域训练，预计优于一般 DNA embedding |
| chromatin/open region | 公共 ATAC/DNase peaks | matched closed regions | species/tissue holdout | AUROC, AUPRC | 若接入标签，长上下文有望优于短窗口 AgroNT |
| expression proxy | promoter/gene body -> expression bin | matched genes | tissue/species holdout | Spearman, AUROC | 需要外部表达标签；长上下文预计优于短上下文模型 |
| variant effect | ref/alt 已知功能变异 | neutral/matched variants | gene/LD block holdout | AUROC, Spearman | causal likelihood + RC consistency，预计优于纯 MLM embedding |

## 12. 基线和预计优势

必须比较:

- CNN/DeepSEA-like 从头训练: 本模型预计在所有小样本和跨物种任务上更好。
- DNABERT-2: 本模型预计在单点变异、splice/TIS/TTS 和长上下文 promoter/TES 上更好。
- AgroNT: AgroNT 是植物强基线；本模型预计在结构注释相关任务、长上下文任务、变异 likelihood 上更好，但在某些短 promoter 分类任务上不保证全面超过。
- PlantCAD/PlantCAD2: 本模型架构接近，但本项目使用 262 个有注释作物 genome、区域加权训练，预计在作物结构区域任务上更有优势；若 PlantCAD2 已用更大植物数据，本模型不保证全任务超过，但应在本地作物注释任务和严格 holdout 上更贴合。
- HyenaDNA/Evo 2: 通用基因组长模型很强；本模型预计在作物结构注释和区域任务上更好，通用生成能力不一定超过。

最有把握超过基线的任务:

1. splice donor/acceptor 跨物种评测。
2. TIS/TTS 和 CDS/UTR/intron 区域分类。
3. gene family holdout 下的 lncRNA/mRNA。
4. 区域加权后的 promoter/TES 分类。

不确定、需要实测的任务:

- expression regression。
- chromatin/open region 跨组织泛化。
- 农艺变异 effect size 排序。

## 13. 进展记录

- 2026-06-07 23:26:04 CST: 读取 `/home/user/zhangzhishuai/data/plantDB/genome/README.md`，确认训练数据口径；完成 AgroNT、PlantCAD/Caduceus、Evo 2、HyenaDNA、DNABERT-2、GROVER 和 DNA foundation benchmark 调研；确定主路线为长上下文、单碱基、RC 等变双向 Mamba/Hyena 模型。
- 2026-06-07 23:46:31 CST: 按用户要求扩展为端到端正式训练方案，补充 `.fna.gz` 扫描、contig QC、split、窗口化、token shard、GPU batch 输入、mask/采样策略、下游监督数据构建，以及 CPU/GPU 资源和每阶段耗时估算。
- 2026-06-08 11:42:31 CST: 按用户要求重构方案，放弃 1644 个无结构注释 genome，正式数据限定为 262 个有 FASTA+GFF/GTF 的 assembly；新增区域加权采样、严格防泄漏 split、片段过滤、跨服务器搬运磁盘估算、详细下游任务和基线优势预期。
