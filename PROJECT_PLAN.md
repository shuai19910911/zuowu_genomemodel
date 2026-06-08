# CropGenome-FM 作物基因组预训练大模型完整训练计划

更新时间: 2026-06-08 19:15:22 CST

## 1. 最终训练口径

本项目第一版正式模型采用“结构注释完整基因组 + 在线采样/tokenization + 小磁盘缓存”的方案。

核心决策:

- 使用 262 个同时具备 genome FASTA 和 GFF3/GTF 的 assembly。
- 放弃 1644 个缺少 GFF3/GTF 结构注释的 genome，不进入第一版预训练。
- 不再预生成全量 `token_shards`、全量 `window_index` 或全量长上下文 shard。
- 训练服务器只搬运原始压缩数据、小索引和配置；训练时在线采样、在线 tokenization。
- 训练服务器只维护 100-200GB 磁盘 cache，循环覆盖，不长期保存历史 shard。

当前正式数据:

| 项目 | 数量/体积 |
|---|---:|
| 有 genome + GFF3/GTF 的 assembly | 262 |
| 覆盖属 | 26 |
| chromosome-level assembly | 233 |
| complete genome assembly | 13 |
| scaffold assembly | 12 |
| contig assembly | 4 |
| genome `.fna.gz` | 213.48 GB |
| GFF/GTF `.gz` | 5.40 GB |
| 原始压缩数据合计 | 218.89 GB |

训练目标: 训练一个结构注释感知、区域加权、长上下文、反向互补一致的作物基因组基础模型 CropGenome-FM。

## 2. 总体技术路线

1. 读取 262 个完整注释 assembly。
2. 流式扫描 FASTA，建立 contig 质量索引。
3. 解析 GFF3/GTF，建立 gene、transcript、exon、CDS、UTR、intron、TSS、TES 区域索引。
4. 构建 promoter、splice flank、TES/polyA、高质量 intergenic 等训练区域。
5. 按 assembly/species/genus/gene-family 严格 split，先 split 后采样，防止泄漏。
6. 在线按区域比例抽取窗口，不全量生成窗口表。
7. 在线读取 `.fna.gz`，生成 token、mask、region_ids、region_weights。
8. 使用 100-200GB 磁盘 cache 保存近期 token/window shard，循环覆盖。
9. 训练 CropGenome-FM-Large: 8K -> 64K -> 128K，资源允许再做 256K。
10. 构建下游任务，比较 CNN、DNABERT-2、AgroNT、PlantCAD2、HyenaDNA/Evo2 等基线。

## 3. 数据预处理

所有脚本、索引、配置和日志都放在当前项目目录下。原始 plantDB 数据目录只读。

### 3.1 目录设计

| 目录 | 内容 | 训练服务器是否需要 | 预计体积 |
|---|---|---|---:|
| `raw_links/` | 指向原始 `.fna.gz/.gff.gz/.gtf.gz` 的路径表或软链接清单 | 是 | < 1 GB |
| `data_manifests/` | assembly manifest、split、属/物种统计 | 是 | < 1 GB |
| `sequence_index/` | contig 长度、GC、N、softmask、header、offset | 是 | 1-5 GB |
| `annotation_index/` | gene、transcript、exon、CDS、UTR、intron、TSS、TES | 是 | 5-30 GB |
| `sampling_index/` | 区域候选池、权重表、过滤规则、split map | 是 | 5-30 GB |
| `token_cache/` | 在线生成的临时 token/window shard | 是，循环覆盖 | 100-200 GB |
| `configs/` | 数据、模型、训练配置 | 是 | < 1 GB |
| `logs/` | Slurm 和训练日志 | 是 | 20-50 GB |
| `checkpoints/` | 最近 checkpoint 和 best inference 权重 | 是 | 100-300 GB |
| `results/` | 下游评测结果 | 是 | 20-100 GB |

不搬运到训练服务器:

- 全量 `window_index`。
- 全量长期 `token_shards`。
- 历史 checkpoint。
- 大量中间 debug 文件。

### 3.2 Assembly manifest

输入:

- `/home/user/zhangzhishuai/data/plantDB/genome/local_reports/completed-genome-index.tsv`

输出:

- `data_manifests/assemblies.tsv`

字段:

- `assembly_id`
- `assembly_accession`
- `species`
- `genus`
- `assembly_level`
- `source`
- `genome_path`
- `gff3_path`
- `gtf_path`
- `genome_size_bytes`
- `annotation_size_bytes`
- `has_gff3`
- `has_gtf`
- `split_group`

过滤:

- 只保留 `has_genome=yes` 且 `has_gff3=yes` 或 `has_gtf=yes`。
- 优先 chromosome/complete genome。
- scaffold/contig assembly 保留，但后续区域和窗口更严格过滤。

资源:

- 8-16 CPU 核。
- 32GB RAM。
- 5-20 分钟。

### 3.3 FASTA 质量扫描

输入:

- 262 个 `.fna.gz`。

处理:

- 流式读取 gzip，不整体解压到磁盘。
- 统计每条 contig/chromosome:
  - length
  - A/C/G/T/N count
  - GC
  - N fraction
  - lowercase/softmask fraction
  - header
  - assembly_id
  - 是否 organelle/plastid/mitochondrial 候选

输出:

- `sequence_index/contigs.tsv`

过滤:

- contig length `< 10 kb` 不进入训练。
- contig N fraction `> 10%` 不进入训练。
- contig N fraction `5%-10%` 只允许低权重非关键区域。
- GC 属内 z-score `>|4|` 标记为异常，低权重或剔除。
- organelle、plastid、mitochondrial 单独标记，第一版不混入核基因组主训练。

资源:

- 2-6 个 CPU 作业。
- 每作业 30 核、100-150GB RAM。
- 8-24 小时。

### 3.4 GFF/GTF 结构注释解析

输入:

- 262 个 assembly 的 GFF3/GTF。

输出:

- `annotation_index/genes.parquet`
- `annotation_index/transcripts.parquet`
- `annotation_index/exons.parquet`
- `annotation_index/cds.parquet`
- `annotation_index/utr.parquet`
- `annotation_index/introns.parquet`

解析内容:

- gene_id
- transcript_id
- feature type
- chromosome/contig
- start/end
- strand
- phase
- biotype
- parent-child relation

质量控制:

- 坐标必须在 contig 长度内。
- gene/transcript/exon/CDS strand 必须一致。
- exon 顺序合法。
- CDS phase 冲突严重的 transcript 不进入 CDS/TIS/TTS/splice 任务。
- 多转录本基因选 canonical transcript 作为主训练区域，其余 isoform 保留标签但低权重。
- protein-coding、lncRNA、pseudogene、TE gene 分开标记。

资源:

- 2-6 个 CPU 作业。
- 每作业 30 核、80-150GB RAM。
- 8-36 小时。

### 3.5 功能区域构建

基于 GFF/GTF 构建训练区域:

| 区域 | 定义 |
|---|---|
| CDS/exon | protein-coding exon 和 CDS 区间 |
| splice donor/acceptor | exon-intron junction 周围窗口 |
| UTR | 5'UTR 和 3'UTR，若注释可用 |
| promoter/TSS | TSS 上游 5kb + 下游 1kb |
| TES/polyA | TES 上下游各 3kb |
| intron | transcript 内 exon 间区间 |
| high-quality intergenic | 远离 gene、低 N、低 repeat、GC 正常、非 gap 区 |
| background | 少量随机 genome 区域，防止模型只见注释区域 |

输出:

- `annotation_index/regions.parquet`
- `sampling_index/region_candidates.parquet`
- `sampling_index/region_weights.tsv`

资源:

- 2-6 个 CPU 作业。
- 每作业 30 核、80-150GB RAM。
- 6-24 小时。

## 4. 片段过滤和采样策略

### 4.1 不全量输入所有序列

训练不是把全部 genome 随机切片后全量输入。每个候选窗口需要满足:

- 有明确 split。
- 属于可接受区域。
- 通过 contig QC。
- 通过 window QC。
- 不和 val/test 发生坐标或相似片段泄漏。
- 满足候选池保留规则，并在训练时满足区域采样比例。

### 4.2 硬质量过滤

所有训练、验证、测试窗口必须先通过硬质量过滤。规则在候选池构建阶段执行，不能等到训练时再临时判断。

N 比例:

- train 默认 `N <= 5%`。
- `5%-10%` 只允许稀缺小属或稀缺功能区域低权重救援，且需要在 sampling index 中记录 `rescued_by_low_abundance=true`。
- validation/test 必须 `N <= 5%`，不做救援。

连续 N:

- 任意连续 `N >= 1 kb` 的窗口丢弃。
- CDS、splice、start/stop 监督窗口中连续 `N >= 100 bp` 丢弃。

有效碱基:

- 普通预训练窗口要求 `A/C/G/T >= 90%`。
- 关键监督窗口要求 `A/C/G/T >= 98%`，包括 CDS frame、splice donor/acceptor、start/stop codon、TSS/TES probe。

低复杂度:

- 单一碱基比例 `> 80%` 的窗口丢弃。
- dust/entropy 标记为低复杂度的纯背景窗口丢弃。
- CDS/splice/start/stop 若局部复杂度偏低但注释可靠，保留监督标签，同时降低 MLM mask 比例并记录低复杂度标记。

contig 边缘:

- 距 contig/scaffold 两端不足 `1 kb` 的窗口默认丢弃。
- 若窗口包含完整 gene model、完整 CDS 或完整 transcript boundary，可作为功能窗口保留，但需要记录 `near_contig_edge=true`，validation/test 中不使用这类边缘窗口。

注释可靠性:

- CDS 坐标越界、transcript parent 缺失、exon/CDS parent 关系断裂的区域不用于监督任务。
- CDS 长度不是 3 的倍数、缺 start/stop 或含内部 stop 的 transcript 不用于 CDS frame/start/stop 监督。
- 这些区域仍可作为普通预训练序列进入候选池，但 `region_id` 不能标为可靠 CDS/splice/frame。

### 4.3 候选窗口区域保留比例

这一步决定哪些窗口进入候选池，用于压缩总数据量。它不是最终 batch 采样比例；最终训练还会从候选池中按 4.5 的目标比例动态采样。

| 区域 | 候选池保留比例 | 保留范围和条件 |
|---|---:|---|
| CDS / coding exon | 100% | 所有 coding exon、CDS frame、start/stop 相关窗口进入候选池 |
| splice donor/acceptor | 100% | donor/acceptor 上下游至少 `+/-2 kb` 进入候选池 |
| start/stop codon neighborhood | 100% | start/stop 上下游至少 `+/-2 kb` 进入候选池 |
| UTR | 100% | 已注释 5UTR/3UTR 全部保留；transcript boundary 周边窗口全部保留 |
| promoter/TSS core | 100% | TSS upstream `0-5 kb` 全部保留 |
| promoter/TSS distal | 15% | TSS upstream `5-20 kb` 只保留高质量代表窗口 |
| weak promoter | 100% of weak window | 无可靠 TSS 时，只用 gene upstream `2 kb` 作为弱 promoter 标签 |
| intron boundary | 100% | exon-intron boundary 两侧 `2 kb` 全部保留 |
| ordinary intron interior | 10% | 普通 intron 内部随机分层保留 |
| long intron interior | 5% | `>20 kb` 长 intron 内部优先保留 GC/复杂度正常、`N <= 2%` 的窗口 |
| TE/repeat annotated | 50% | 只有可靠 repeat 注释时启用 |
| gene-proximal TE/repeat | 100% | 距 gene body 或 promoter `20 kb` 内的 TE/repeat 全部保留 |
| TE boundary | 100% | TE 边界上下游 `+/-2 kb` 全部保留 |
| gene-proximal intergenic | 10% | 距任意 gene `20 kb` 内，优先 `N <= 2%`、非低复杂度、完整覆盖窗口 |
| distal intergenic / far noncoding | 3%-5% | `N <= 2%`、无长 N、非低复杂度、非高度重复、GC 在本 genome `5%-95%` 分位范围内 |
| random genome coverage | 1%-2% | 从通过 hard filter 的全基因组窗口中额外抽样，避免完全丢失背景分布 |

TE/repeat 规则:

- 有 repeat 注释的 assembly 才启用 TE/repeat 候选池。
- 无 repeat 注释 genome 不把 intergenic 伪标为 non-repeat 或 TE/repeat。
- TE/repeat 只作为区域标签和采样层，不覆盖 CDS、splice、UTR、promoter 等高优先级功能标签。

### 4.4 去冗余和代表性控制

为了进一步降低存储和重复学习，在同一个 split 内执行去冗余；不同 split 之间先做防泄漏相似性检查，不能通过去冗余掩盖泄漏。

- 同一 assembly 内高度相似窗口按 minimizer/simhash 去冗余。
- 对普通 intergenic 和 repeat-rich 背景，若窗口相似度 `>= 95%`，只保留 1 个代表。
- 对 CDS、splice、start/stop 不做相似性丢弃，只做质量过滤。
- 每个 assembly 的 distal intergenic token 占比不超过该 assembly 训练 token 的 `5%`。
- 每个属的 ordinary intergenic token 占比不超过该属训练 token 的 `10%`。
- 稀缺小属的功能区域不得被全局去冗余过度压缩；CDS/splice/start/stop 至少保留到可支撑该属下游 probe 的规模。

### 4.5 训练 batch 区域采样比例

区域采样参考 `douke_genome` 项目的思路: 优先保留 CDS、splice、promoter/TSS、UTR 等功能区和边界区；intron/intergenic 降低比例；TE/repeat 只有在存在可靠 TE/repeat 注释时进入，不允许把普通 intergenic 伪标为 non-repeat 或 TE。参考: https://github.com/shuai19910911/douke_genome/blob/main/PLAN.md

训练 batch 的目标区域比例分为两种模式。

模式 A: 有可靠 TE/repeat 注释时使用。

| 区域 | 采样比例 | loss 权重 | 目的 |
|---|---:|---:|---|
| CDS/protein-coding exon | 25% | 1.50 | 密码子、ORF、保守编码结构 |
| splice donor/acceptor | 15% | 2.00 | 剪接位点和外显子边界 |
| promoter/TSS | 15% | 1.40 | 启动子和表达调控 |
| UTR | 10% | 1.20 | 翻译调控、mRNA 稳定性 |
| TES/polyA | 5% | 1.20 | 转录终止和 polyA |
| intron | 10% | 0.90 | 长程 gene body 和调控上下文 |
| TE/repeat annotated | 12% | 0.80 | 作物基因组重复序列背景和调控相关重复 |
| high-quality intergenic | 5% | 0.60 | 高质量非编码背景 |
| random background | 3% | 0.50 | 保留 genome-wide 分布 |

模式 B: 当前无可靠 TE/repeat 注释时使用。TE/repeat 的 12% 不启用，重新分配给 CDS、splice、TES、intron 和高质量 intergenic。

| 区域 | 采样比例 | loss 权重 | 目的 |
|---|---:|---:|---|
| CDS/protein-coding exon | 28% | 1.50 | 密码子、ORF、保守编码结构 |
| splice donor/acceptor | 18% | 2.00 | 剪接位点和外显子边界 |
| promoter/TSS | 15% | 1.40 | 启动子和表达调控 |
| UTR | 10% | 1.20 | 翻译调控、mRNA 稳定性 |
| TES/polyA | 7% | 1.20 | 转录终止和 polyA |
| intron | 12% | 0.90 | 长程 gene body 和调控上下文 |
| high-quality intergenic | 7% | 0.60 | 高质量非编码背景 |
| random background | 3% | 0.50 | 保留 genome-wide 分布 |

这是 batch sampler 的目标比例，不是 genome 的真实比例。每个 epoch 记录实际采样比例并做偏差修正。

当前默认采用模式 B，除非后续为 262 个 assembly 补齐可靠 TE/repeat 注释或外部 repeat annotation。

## 5. 严格防泄漏 split

原则: 一个基因组片段不能同时出现在训练集和验证/测试集。

### 5.1 预训练 split

执行顺序:

1. 先按 assembly/species/genus 分组。
2. 再确定 train/val/test/holdout。
3. 最后在每个 split 内独立采样窗口。

规则:

- 同一 assembly 的所有窗口只能属于一个 split。
- 若 species 有多个 assembly，可设置 species holdout。
- 选择部分小属或重要作物属做 genus holdout。
- val/test 中的 assembly 不允许在 train 中出现任何窗口。
- 若同 assembly 内必须局部切分，train 与 val/test 之间至少保留 `2 x max_context` 坐标间隔；第一版优先避免同 assembly 内切分。

### 5.2 下游 split

- gene-level split: 同一 gene_id 的所有 promoter、CDS、UTR、splice、TIS/TTS 片段不能跨 split。
- gene-family split: 高相似 paralog family 整体分到同一 split。
- species/genus holdout: 检查跨物种和跨属迁移能力。
- variant split: 同一 variant、同一 LD block、同一 gene 周围窗口不能跨 split。
- random split 只作辅助，不作为主要报告结论。

## 6. 训练服务器存储策略

采用低磁盘策略:

- 搬原始压缩数据。
- 搬轻量索引。
- 不搬全量 token shards。
- 不搬全量 window index。
- 训练时在线采样和在线 tokenization。
- `token_cache/` 限制为 100-200GB，循环覆盖。
- 只保留最近 1 个完整 checkpoint 和 1 个 best inference checkpoint。

### 6.1 训练服务器磁盘需求

| 项目 | 体积 |
|---|---:|
| 原始压缩数据 | 218.89 GB |
| 轻量 manifest + sequence index | 1-5 GB |
| annotation/sampling index | 10-60 GB |
| token/window cache | 100-200 GB |
| logs | 20-50 GB |
| 当前 checkpoint + best model | 100-300 GB |
| results/downstream cache | 20-100 GB |
| 安全余量 | 100-200 GB |

结论:

- 最低可运行: 800GB-1TB 可用磁盘。
- 推荐: 1.5TB 可用磁盘。
- 更稳妥: 2TB 可用磁盘。

### 6.2 训练服务器 CPU/RAM

推荐:

- CPU: 32-64 核。
- RAM: 128-256GB。
- 本地 SSD 优先放 `token_cache/` 和 checkpoint。

最低:

- CPU: 24-32 核。
- RAM: 96-128GB。

## 7. 模型输入

每个训练样本在线生成:

```text
input_ids:       [B, L] int64
labels_mlm:      [B, L] int64, 非 mask 位点为 -100
loss_mask:       [B, L] bool
region_ids:      [B, L] uint8
region_weights:  [B, L] fp16/bf16
quality_scores:  [B, L] optional fp16
rc_flag:         [B] bool
metadata:        assembly_id, species_id, genus_id, contig_id, start, end, strand, split
```

token vocabulary:

| token | id |
|---|---:|
| A | 0 |
| C | 1 |
| G | 2 |
| T | 3 |
| N | 4 |
| MASK | 5 |
| PAD | 6 |
| BOS | 7 |
| EOS | 8 |

IUPAC ambiguous bases 默认转为 N。N、PAD、低质量碱基不参与主 loss。

context:

- Stage B: 8192。
- Stage C1: 65536。
- Stage C2: 131072。
- Stage D: 262144，资源充足后再执行。

## 8. 模型架构

主模型: CropGenome-FM-Large。

架构:

- single-base token embedding。
- region embedding。
- optional strand/quality embedding。
- RC-equivariant bidirectional Mamba2/Hyena backbone。
- 每 4-6 层插入 local/global sparse attention。
- RMSNorm。
- SwiGLU/gated MLP。
- MLM head。
- causal auxiliary head。
- region-aware probe/contrastive head。

推荐配置:

| 项 | 值 |
|---|---:|
| layers | 32 |
| hidden | 1024 |
| MLP ratio | 4 |
| attention interval | every 4 or 6 layers |
| 参数量 | 300M-450M |
| precision | bf16 |
| optimizer | AdamW |
| distributed | FSDP 或 DeepSpeed ZeRO-3 |

备选:

- Base: 100M-150M，资源不足时的正式小一档模型。
- XL: 800M-1.2B，8-16 张 80GB GPU 稳定后再考虑。

## 9. Loss

总 loss:

```text
L_total =
  1.00 * L_region_weighted_MLM
  + 0.10 * L_causal_next_token
  + 0.05 * L_reverse_complement_consistency
  + 0.05 * L_region_contrastive_optional
```

### 9.1 Region-weighted MLM

- mask rate: 15%。
- span length: 1-512 bp。
- CDS、splice、TSS 区域增加短 span。
- intron、intergenic 混合长 span。
- 按 `region_weights` 加权 cross entropy。
- N/PAD/低质量位点不计入 loss。

### 9.2 Causal next-token auxiliary

目的:

- 提供 ref/alt likelihood score。
- 提升变异效应评分能力。

权重低，只在部分 batch 或长上下文阶段启用。

### 9.3 Reverse-complement consistency

同一窗口 forward 与 reverse-complement 的 embedding、masked logits 或 variant score 应一致，用于降低 DNA 方向偏置。

## 10. 训练阶段、资源和时间

### 10.1 CPU 数据处理阶段

| 阶段 | 内容 | 资源 | 时间 |
|---|---|---|---:|
| P1 | assembly manifest | 8-16 核，32GB | < 2 小时 |
| P2 | FASTA QC 扫描 | 2-6 作业，每作业 30 核，100-150GB | 8-24 小时 |
| P3 | GFF/GTF 解析 | 2-6 作业，每作业 30 核，80-150GB | 8-36 小时 |
| P4 | 区域构建和候选池 | 2-6 作业，每作业 30 核，80-150GB | 6-24 小时 |
| P5 | split 和防泄漏检查 | 8-16 核，32-64GB | 2-8 小时 |
| P6 | online sampler dry-run | 1-2 GPU + 16-32 CPU 核 | 2-8 小时 |

CPU 总体: 2-5 天。

### 10.2 GPU 预训练阶段

| 阶段 | context | token 预算 | 推荐 GPU | 预计时间 | 目标 |
|---|---:|---:|---:|---:|---|
| Stage B | 8K | 30B-80B | 4-8 x 80GB | 5-18 天 | 局部 motif、CDS、splice、TSS/TES |
| Stage C1 | 64K | 15B-40B | 8 x 80GB | 7-18 天 | gene body、promoter-gene、长 intron |
| Stage C2 | 128K | 5B-20B | 8 x 80GB | 5-14 天 | 远端调控和结构上下文 |
| Stage D | 256K | 2B-10B | 8-16 x 80GB | 4-14 天 | 资源允许后做长上下文 midtraining |

推荐训练服务器:

- 磁盘: 1.5TB 可用空间，最低 800GB-1TB。
- CPU: 32-64 核。
- RAM: 128-256GB。
- GPU: 推荐 4-8 张 80GB；最低 2 张 80GB 可跑 Base 或 Large 8K/64K。

总体时间:

- CPU 预处理: 2-5 天。
- 8K + 64K/128K 正式训练: 4-8 周。
- 下游评测: 1-2 周。
- 完整第一版报告模型: 6-10 周。

## 11. 下游任务

下游任务只使用结构注释完整 assembly 和外部公开功能组数据。所有任务必须使用严格 split。

| 任务 | 正例 | 负例 | split | 指标 | 预期优势 |
|---|---|---|---|---|---|
| splice donor/acceptor | exon-intron junction | 同 contig 非 junction GT/AG + random | gene family + species holdout | AUROC, AUPRC, F1 | splice 高采样和高 loss 权重，预计优于 DNABERT-2/AgroNT |
| TIS/TTS | CDS start/stop codon 周围 | 同 frame 非起止 codon | gene family holdout | AUROC, AUPRC | CDS 高权重 + 单碱基建模，预计优于 k-mer 模型 |
| promoter/TSS | TSS 上游 5kb + 下游 1kb | matched intergenic | assembly/species holdout | AUROC, AUPRC | 长上下文 + TSS 权重，预计优于短窗口 CNN/DNABERT-2 |
| TES/polyA | TES 周围窗口 | matched downstream negatives | gene holdout | AUROC, AUPRC | TES 专门采样，预计优于未做区域感知预训练模型 |
| CDS/UTR/intron 分类 | 注释区域 | matched background | assembly holdout | macro-F1 | 区域 embedding 直接支持，预计明显优于 CNN |
| lncRNA/mRNA | transcript 注释 | 长度/GC 匹配负例 | species holdout | AUROC, MCC | transcript 区域训练，预计优于通用 DNA embedding |
| chromatin/open region | ATAC/DNase peaks | matched closed regions | species/tissue holdout | AUROC, AUPRC | 接入标签后，长上下文可能优于短窗口模型 |
| expression proxy | promoter/gene body -> expression bin | matched genes | tissue/species holdout | Spearman, AUROC | 依赖外部表达标签，作为中后期任务 |
| variant effect | ref/alt 功能变异 | neutral/matched variants | gene/LD block holdout | AUROC, Spearman | causal likelihood + RC consistency，预计优于纯 MLM embedding |

## 12. 基线比较

必须比较:

- CNN/DeepSEA-like 从头训练。
- DNABERT-2。
- AgroNT。
- PlantCAD/PlantCAD2。
- HyenaDNA/Evo 2 可用 checkpoint 或 embedding/zero-shot score。

预期最有把握超过基线的任务:

1. splice donor/acceptor。
2. TIS/TTS。
3. CDS/UTR/intron 区域分类。
4. promoter/TES 严格 holdout。
5. gene-family holdout 下的 lncRNA/mRNA。

不保证全面超过的任务:

- expression regression。
- chromatin/open region 跨组织泛化。
- 农艺变异 effect size 排序。

所有“优于基线”的说法必须以后续严格 benchmark 为准。

## 13. 评测结果记录与更新规则

评测结果必须写回本计划文档，GitHub 只记录摘要和关键表格，不上传大规模预测文件、embedding、逐样本结果或训练 checkpoint。详细结果保留在本地 `results/`，计划文档记录可复现路径、指标和结论。

### 13.1 结果文件存储

本地结果建议结构:

```text
results/
  pretrain/
    stage_b_8k_metrics.tsv
    stage_c1_64k_metrics.tsv
    stage_c2_128k_metrics.tsv
  downstream/
    splice_donor_acceptor/
    tis_tts/
    promoter_tss/
    tes_polya/
    region_classification/
    lncrna_mrna/
    chromatin_open_region/
    expression_proxy/
    variant_effect/
  baselines/
    cnn/
    dnabert2/
    agront/
    plantcad2/
    hyenadna_evo2/
```

GitHub 文档只记录:

- 训练阶段。
- checkpoint 名称或编号。
- 数据 split 名称。
- 任务名。
- 指标。
- 基线对照。
- 主要结论。
- 本地结果路径。

### 13.2 预训练结果记录表

| 更新时间 | 阶段 | context | token 数 | train loss | val loss | RC consistency | tokens/s | GPU | checkpoint | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 待填 | Stage B | 8K | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 待填 | Stage C1 | 64K | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 待填 | Stage C2 | 128K | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 待填 | Stage D | 256K | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

预训练阶段必须记录:

- train/val loss。
- 按区域分组 loss: CDS、splice、promoter/TSS、TES、UTR、intron、intergenic。
- 按属/物种分组 val loss。
- RC consistency 指标。
- tokens/s 和 GPU 显存峰值。
- cache 命中率和在线 tokenization 吞吐。

### 13.3 下游评测结果总表

| 更新时间 | 任务 | split | 模型 | AUROC | AUPRC | F1/MCC | 其他指标 | 最强基线 | 是否超过基线 | 本地结果路径 | 结论 |
|---|---|---|---|---:|---:|---:|---|---|---|---|---|
| 待填 | splice donor/acceptor | gene-family + species holdout | CropGenome-FM | 待填 | 待填 | 待填 | - | 待填 | 待填 | 待填 | 待填 |
| 待填 | TIS/TTS | gene-family holdout | CropGenome-FM | 待填 | 待填 | 待填 | - | 待填 | 待填 | 待填 | 待填 |
| 待填 | promoter/TSS | assembly/species holdout | CropGenome-FM | 待填 | 待填 | 待填 | - | 待填 | 待填 | 待填 | 待填 |
| 待填 | TES/polyA | gene holdout | CropGenome-FM | 待填 | 待填 | 待填 | - | 待填 | 待填 | 待填 | 待填 |
| 待填 | CDS/UTR/intron classification | assembly holdout | CropGenome-FM | 待填 | 待填 | 待填 | macro-F1 待填 | 待填 | 待填 | 待填 | 待填 |
| 待填 | lncRNA/mRNA | species holdout | CropGenome-FM | 待填 | 待填 | 待填 | MCC 待填 | 待填 | 待填 | 待填 | 待填 |
| 待填 | chromatin/open region | species/tissue holdout | CropGenome-FM | 待填 | 待填 | 待填 | - | 待填 | 待填 | 待填 | 待填 |
| 待填 | expression proxy | tissue/species holdout | CropGenome-FM | 待填 | 待填 | 待填 | Spearman 待填 | 待填 | 待填 | 待填 | 待填 |
| 待填 | variant effect | gene/LD block holdout | CropGenome-FM | 待填 | 待填 | 待填 | Spearman 待填 | 待填 | 待填 | 待填 | 待填 |

### 13.4 基线比较记录表

每个下游任务至少记录以下基线:

| 任务 | CNN/DeepSEA-like | DNABERT-2 | AgroNT | PlantCAD2 | HyenaDNA/Evo2 | CropGenome-FM | 当前结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| splice donor/acceptor | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| TIS/TTS | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| promoter/TSS | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| TES/polyA | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| CDS/UTR/intron | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| lncRNA/mRNA | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| variant effect | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

### 13.5 结果解释规则

- 只有严格 holdout split 上超过基线，才写“超过基线”。
- random split 结果只能作为辅助，不作为主结论。
- 若只在某些任务超过，应写“在某些任务上超过”，不能写“全面超过”。
- 若结果不稳定，必须记录方差、重复次数和失败原因。
- 若更换数据、split、模型参数或 checkpoint，结果表必须新增一行，不覆盖旧结果。
- 每完成一个阶段，README 的“项目进展”只写一句摘要，详细表格保留在本节。

当前评测状态:

- 预训练尚未开始。
- 下游评测尚未开始。
- 所有结果表为预注册模板，后续训练和评测完成后逐项填写。

## 14. 执行顺序

近期执行顺序:

1. 生成 `assemblies.tsv`。
2. 生成 `contigs.tsv`。
3. 解析 GFF/GTF，生成 annotation index。
4. 构建 region candidates 和 sampling index。
5. 生成 split，并跑防泄漏检查。
6. 实现 online sampler 和 tokenization。
7. 在训练服务器跑 1000 step dry-run，校正吞吐和缓存策略。
8. 启动 Stage B 8K 正式预训练。
9. 完成 Stage B 后做第一批下游 probe。
10. 进入 64K/128K 继续预训练。

## 15. 进展记录

- 2026-06-07 23:26:04 CST: 读取 `/home/user/zhangzhishuai/data/plantDB/genome/README.md`，确认训练数据口径；完成 AgroNT、PlantCAD/Caduceus、Evo 2、HyenaDNA、DNABERT-2、GROVER 和 DNA foundation benchmark 调研；确定主路线为长上下文、单碱基、RC 等变双向 Mamba/Hyena 模型。
- 2026-06-07 23:46:31 CST: 按用户要求扩展为端到端正式训练方案，补充 `.fna.gz` 扫描、contig QC、split、窗口化、token shard、GPU batch 输入、mask/采样策略、下游监督数据构建，以及 CPU/GPU 资源和每阶段耗时估算。
- 2026-06-08 11:42:31 CST: 按用户要求重构方案，放弃 1644 个无结构注释 genome，正式数据限定为 262 个有 FASTA+GFF/GTF 的 assembly；新增区域加权采样、严格防泄漏 split、片段过滤、跨服务器搬运磁盘估算、详细下游任务和基线优势预期。
- 2026-06-08 15:00:51 CST: 根据用户确认，放弃进一步压缩到核心 assembly 的方案，整理为最终完整训练计划: 262 个结构注释完整基因组 + 原始压缩数据搬运 + 小索引 + 在线采样/tokenization + 100-200GB 磁盘缓存。
- 2026-06-08 18:16:18 CST: 按用户要求新增评测结果记录规范，预置预训练指标表、下游任务结果表、基线比较表和结果解释规则；明确 GitHub 只记录摘要和关键表格，不上传大结果文件。
- 2026-06-08 18:45:35 CST: 按用户要求参考 `douke_genome` 的区域采样方案，调整为“有 TE/repeat 注释模式”和“无可靠 TE/repeat 注释 fallback 模式”；当前默认使用无 TE fallback，避免把普通 intergenic 伪标为 repeat/non-repeat。
- 2026-06-08 19:15:22 CST: 按用户给定的 4.5.1-4.5.3 规则重构输入处理: 增加硬质量过滤、候选池区域保留比例、minimizer/simhash 去冗余和 assembly/genus 级 intergenic token 上限；明确候选池保留比例不同于训练 batch 采样比例。
