# CropGenome-FM 作物基因组预训练大模型完整训练计划

更新时间: 2026-06-11 08:20:04 CST

## 1. 最终训练口径

本项目第一版正式模型采用“结构注释完整基因组 + 本服务器固化每个 stage 的训练输入 + 训练服务器动态 mask/label/RC 训练”的方案。

核心决策:

- 使用 263 行同时具备 genome FASTA 和 GFF3/GTF 的 crop assembly manifest；按 assembly accession 去重后为 258 个 canonical assembly accession。
- 放弃 1644 个缺少 GFF3/GTF 结构注释的 genome，不进入第一版预训练。
- 不预生成固定 mask、固定 MLM label、固定 batch 顺序或固定 RC 增强。
- 在本服务器完成数据处理、候选池构建、split、防泄漏检查，并按 Stage B/C1/C2/D 一次性固化该 stage 的训练输入窗口或 `input_ids`。
- 训练服务器只搬运 `training_server_transfer/` 数据、配置和小元数据；训练时动态生成 mask、labels、RC 增强和 batch 顺序。
- 当前正在按“主 context 长度全部保留 + 其他长度受控回放”的新策略重新生成 Stage B/C1/C2/D 固化输入；如训练服务器磁盘紧张，也可以只保留当前要训练的一个 stage 子目录。

当前正式数据:

| 项目 | 数量/体积 |
|---|---:|
| 有 genome + GFF3/GTF 的 crop manifest 行 | 263 |
| canonical assembly accession | 258 |
| 覆盖属 | 26 |
| canonical split train/val/test | 192/35/31 |
| 训练服务器最终搬运目录 | `training_server_transfer/` |
| 最终搬运目录体积 | 67G (`du -sh training_server_transfer`, 2026-06-10 23:47:01 CST) |
| 搬运目录 SHA256 | 已刷新并通过 `sha256sum -c SHA256SUMS` |

训练目标: 训练一个结构注释感知、区域加权、长上下文、反向互补一致的作物基因组基础模型 CropGenome-FM。

## 2. 总体技术路线

1. 本服务器读取 263 行完整注释 crop assembly manifest，并折叠为 258 个 canonical assembly accession。
2. 本服务器流式扫描 FASTA，建立 contig 质量索引。
3. 本服务器解析 GFF3/GTF，建立 gene、transcript、exon、CDS、UTR、intron、TSS、TES 区域索引。
4. 本服务器构建 promoter、splice flank、TES/polyA、高质量 intergenic 等训练区域，并在可靠证据存在时增加端粒、着丝粒/pericentromere、TE/repeat、satellite/tandem repeat、rDNA、organellar insertion、segmental duplication 和 synteny-breakpoint 等结构基因组区域。
5. 本服务器按 assembly/species/genus/gene-family 严格 split，先 split 后采样，防止泄漏。
6. 本服务器按 4.2-4.5 规则构建候选池、去冗余、区域采样和 context bucket 配方。
7. 本服务器按 Stage B/C1/C2/D 分别一次性固化该 stage 的输入窗口或 `input_ids`，但不固化 mask/label/batch order/RC。
8. 将 `training_server_transfer/` 中的 stage 数据、配置、小元数据和 manifest 传输到训练服务器。
9. 训练服务器读取固化输入，训练时动态 mask、动态 MLM label、动态 RC 增强、动态 batch 顺序。
10. 训练 CropGenome-FM-Large: 8K -> 64K -> 128K，资源允许再做 256K。
11. 构建下游任务，比较 CNN、DNABERT-2、AgroNT、PlantCAD2、HyenaDNA/Evo2 等基线。

## 3. 数据预处理

所有脚本、索引、配置和日志都放在当前项目目录下。原始 plantDB 数据目录只读。

### 3.1 目录设计

| 目录 | 内容 | 训练服务器是否需要 | 预计体积 |
|---|---|---|---:|
| `raw_links/` | 指向原始 `.fna.gz/.gff.gz/.gtf.gz` 的路径表或软链接清单，仅本服务器使用 | 否 | < 1 GB |
| `data_manifests/` | assembly manifest、split、属/物种统计，本服务器中间目录 | 否，最终只搬运 `metadata/` 摘要 | < 1 GB |
| `sequence_index/` | contig 长度、GC、N、softmask、header、offset，本服务器中间目录 | 否，最终只搬运 QC 摘要 | 约 2.2 GB |
| `annotation_index/` | gene、transcript、exon、CDS、UTR、TSS/TES、可选结构基因组注释，本服务器中间目录 | 否，最终只搬运 QC 摘要 | 约 47 GB，加入 repeat/centromere/telomere 后可能增加 |
| `sampling_index/` | 区域候选池、权重表、过滤规则、split map，本服务器中间目录 | 否，最终只搬运候选汇总 | 约 7.3 GB |
| `training_server_transfer/` | 专门给用户传到训练服务器的简约目录，包含 configs、metadata、inputs、manifest 和 SHA256 | 是 | 约 67 GB |
| `configs/` | 数据、模型、训练配置 | 是 | < 1 GB |
| `logs/` | 本服务器预处理日志和训练服务器训练日志 | 可选 | 20-50 GB |
| `checkpoints/` | 最近 checkpoint 和 best inference 权重，训练服务器产生 | 不从本服务器搬运 | 100-300 GB |
| `results/` | 下游评测结果，训练服务器或评测服务器产生 | 不从本服务器搬运 | 20-100 GB |

`training_server_transfer/` 是唯一推荐用户整体传输到训练服务器的目录。它的内容由本服务器准备，不上传大文件到 GitHub。GitHub 只保留该目录的说明文件和计划文档。

不搬运到训练服务器:

- 原始 plantDB 全量数据，除非后续决定训练服务器重新取序列。
- 本服务器中间 debug 文件。
- 已完成且不再训练的旧 stage 输入。
- 历史 checkpoint。
- 下游大结果文件。

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

- 263 行 crop manifest 对应的 `.fna.gz`，canonical accession 去重后用于 split 防泄漏。

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

- 263 行 crop manifest 对应的 GFF3/GTF。

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

### 3.5.1 结构基因组增强区域

除常规基因结构区域外，后续版本增加可选的 structural-genome annotation layer。该层不替代 CDS、splice、promoter 等功能标签，而是在可靠证据存在时作为额外 `region_bucket`、`structure_flags`、`repeat_family_id` 和下游 probe 来源。

当前 258 个 canonical assembly 的结构注释可行性初评见 [STRUCTURAL_ANNOTATION_FEASIBILITY.md](STRUCTURAL_ANNOTATION_FEASIBILITY.md)。基于现有 contig/QC 元数据，第一批建议:

- 228 个 assembly 作为 TE + telomere + centromere + satellite 全套候选。
- 236 个 assembly 可做 TE/repeat 和 satellite/tandem repeat 自注释。
- 229 个 assembly 可做 telomere/subtelomere 候选和 centromere/pericentromere 候选。
- 21 个 assembly 暂不做结构增强，只保留已有 gene/CDS/splice/promoter/TES 等注释训练。

| 区域 | 定义和来源 | 第一版处理口径 |
|---|---|---|
| telomere | contig/scaffold 端部富集植物端粒 motif，如 `TTTAGGG/CCCTAAA` 串联重复；优先使用 T2T assembly 或显式 telomere 注释 | contig 末端 `0-50 kb` 内检测 motif density 和 tandem repeat；只有 motif 密度达阈值时标为 telomere，不把普通 contig edge 伪标为 telomere |
| subtelomere | telomere 内侧 `50 kb-500 kb` 或物种自适应长度区域，常含 TE、快速演化基因和结构变异 | 作为长上下文区域保留，重点用于 64K/128K/256K 阶段 |
| centromere core | CENH3/Hi-C/遗传图谱/T2T 注释支持的 centromere 区域，或明显 centromeric satellite 阵列 | 只有外部注释或强 satellite 阵列证据时启用；低置信候选只标为 centromere_like，不进入严格监督标签 |
| pericentromere | centromere 两侧 repeat-rich、低重组、TE-rich 区域 | 若无明确边界，使用 centromere core 上下游 `0.5-2 Mb` 或 species-specific 边界，低权重进入长上下文训练 |
| TE/repeat family | RepeatMasker/EDTA/RepeatModeler/现成 repeat GFF 注释；区分 LTR/Gypsy/Copia/LINE/SINE/DNA transposon/Helitron 等 | 有可靠 repeat annotation 才启用；TE family 作为 `repeat_family_id`，不覆盖功能区主标签 |
| TE insertion boundary | TE interval 起止点上下游 `+/-2 kb` | 100% 保留高质量边界，用于学习 TE 插入、调控边界和结构变异信号 |
| intact LTR candidate | 完整 LTR retrotransposon 或 LTR pair 注释 | 若有 EDTA/LTRharvest 结果，单独作为 TE 子任务；否则不伪造 |
| satellite/tandem repeat | TRF/ULTRA/RepeatMasker 标注的 tandem repeat 和 satellite repeat | 区分低复杂度垃圾窗口与高置信 satellite；高置信 satellite 可用于 centromere/telomere 辅助标签 |
| rDNA/organellar insertion | 45S/5S rDNA cluster、chloroplast/mitochondrial insertion 到核基因组片段 | 作为特殊重复/结构区域，避免误当普通 gene-proximal intergenic |
| segmental duplication | self-alignment/minimap2/MCScan 或已有注释支持的大段重复 | 高相似重复区用于去冗余和变异/拷贝数相关下游任务 |
| synteny breakpoint | 跨物种共线性断点、倒位/易位边界 | 作为后期比较基因组增强任务；第一版只预留字段，不强制生成 |

结构基因组层的证据等级:

| 等级 | 证据 | 可用于 |
|---|---|---|
| high | 外部 GFF/BED 注释、T2T 注释、CENH3/Hi-C/遗传图谱、EDTA/RepeatMasker family 注释一致 | 候选池保留、region label、辅助监督和下游任务 |
| medium | motif density、tandem repeat、GC/coverage/softmask pattern 等序列证据强，但缺少外部实验证据 | 候选池保留和弱标签；不用于严格监督 benchmark |
| low | 只有 contig 位置或低复杂度特征 | 只能作为过滤/质量 flag，不作为 telomere/centromere/TE 正例 |

关键原则:

- 没有结构注释的 assembly 仍可用于 gene/CDS/splice 等已可靠区域，但不启用 TE/centromere/telomere 监督标签。
- 端粒和着丝粒在非 T2T assembly 中容易缺失或断裂；只在证据达标时使用，避免把 scaffold 边缘或 gap 区误标。
- TE/repeat 需要 family-level 注释才进入 TE family 子任务；只有 softmask 而无 family 时只作为 repeat-rich flag。
- 结构区域优先服务长上下文阶段 C1/C2/D，不应挤占 CDS/splice 等高置信核心功能 token。

输出:

- `annotation_index/regions.parquet`
- `sampling_index/region_candidates.parquet`
- `sampling_index/region_weights.tsv`

资源:

- 2-6 个 CPU 作业。
- 每作业 30 核、80-150GB RAM。
- 6-24 小时。

### 3.6 Stage 输入固化与训练服务器传输目录

当前正式方案不是训练服务器完全在线采样，也不是提前固化完整监督样本，而是在本服务器按 stage 固化训练输入。

固化内容:

- `input_ids` 或原始 sequence window，优先 `uint8 input_ids`。
- `window_id`。
- `assembly_id`、`species_id`、`genus_id`。
- `contig_id`、`start`、`end`、`strand`。
- `split`。
- `context_bucket`: `4K`, `8K`, `16K`, `32K`, `64K`, `128K`, `256K`。
- `region_bucket`: CDS、splice、TSS、UTR、TES、intron、TE/repeat、intergenic、background。
- `region_weight_base`。
- `quality_flags`: N fraction、low-complexity、near-contig-edge、rescued_by_low_abundance 等。

不固化内容:

- MLM mask 位置。
- `labels_mlm`。
- causal loss 位置。
- RC 增强方向和 RC pair 抽样。
- batch 顺序。
- dropout 或训练时增强。
- 动态 loss 微调。

每个 stage 的输入数据按一个完整 stage 逻辑生成，物理上拆成多个 shard，方便校验、搬运和断点续传。

| Stage | 新固化策略 | 实际写入窗口 | 实际写入 token | 实际目录大小 |
|---|---|---:|---:|---:|
| Stage B | 全部 8K 主候选 + 4K/16K 受控 warm-up/replay | 5,594,781 | 41,242,505,216 | 39GB |
| Stage C1 | 全部 64K 主候选 + 4K/8K/16K/32K 受控 replay | 779,304 | 20,470,165,504 | 20GB |
| Stage C2 | 全部 128K 主候选 + 8K/16K/64K 受控 replay | 101,293 | 6,898,204,672 | 6.5GB |
| Stage D | 全部 256K 主候选 + 8K/64K/128K 受控 replay | 18,400 | 2,777,841,664 | 2.6GB |

这里的“主候选全部保留”指: 在 `stage_windows/<Stage>/windows.candidates.tsv.gz` 中已经通过硬质量过滤、区域保留比例、去冗余和 split 防泄漏的候选窗口里，每个 stage 的主 context 长度不再按旧 token 预算截断。Stage B 的主 context 是 8K，因此全部 8K 候选都写入 Stage B；Stage C1 全部写入 64K 候选；Stage C2 全部写入 128K 候选；Stage D 全部写入 256K 候选。其他长度仍是辅助 replay/warm-up，继续按 stage 配方控制，避免短上下文或辅助上下文把主训练目标淹没。

按收紧后的候选生成和编码过滤，四个 stage 实际合计 `71,388,717,056` token，训练服务器搬运目录为 `67G`。这个结果不是旧版固定 30B/15B/5B/2B token 预算，也不是原始 genome 全量窗口；它来自“主 context 合格候选全部保留 + 辅助长度受控 replay/warm-up”的实际写盘结果。计算方式是读取每个 `training_server_transfer/inputs/<Stage>/summary.tsv` 的 `written_tokens` 字段，或从对应 `manifest.tsv` 对每个 shard 的 `tokens` 求和复现。

当前 Stage B 的 `41,242,505,216` token 来源:

1. 先生成 Stage B 候选池，候选必须通过结构注释完整、split 防泄漏、硬质量过滤、区域保留比例和去冗余。
2. Stage B 的主 context 是 8K；所有通过过滤的 8K 主候选都写入，而不是再按 30B 预算截断。
3. 4K 和 16K 仍作为辅助 replay/warm-up，按配置受控补充，用于保留短 motif/CDS/splice 信号并提前接触稍长上下文。
4. 写入时按完整窗口取整，并同时包含 train/validation/test split，所以最终 token 数由实际候选、过滤失败、quota 达成和完整窗口长度共同决定。
5. 最终值记录在 `training_server_transfer/inputs/Stage_B/summary.tsv`: `written_tokens=41242505216`，`written_windows=5594781`。

Stage B 的实际 token 构成:

| 组成 | 实际 token |
|---|---:|
| train split | 40,435,101,696 |
| validation split | 403,083,264 |
| test split | 404,320,256 |
| 合计 | 41,242,505,216 |

按公式写就是:

```text
Stage_B_actual_tokens
  = sum(Stage_B manifest.tsv 中每个 shard 的 tokens)
  = train_tokens + validation_tokens + test_tokens
  = 40,435,101,696 + 403,083,264 + 404,320,256
  = 41,242,505,216
```

因此需要区分三类比例:

| 比例类型 | 作用位置 | 例子 | 含义 |
|---|---|---|---|
| 候选池区域保留比例 | 本服务器预处理阶段 | distal intergenic 只保留 3%-5% | 决定哪些原始窗口有资格进入候选池，用于压缩数据和去掉低价值背景 |
| stage 长度 token 比例 | 本服务器 stage 固化阶段 | Stage B 中 70% token 来自 8K | 决定一个 stage 内不同 context length 对总 token 的贡献 |
| 训练 batch 区域采样比例 | stage 固化和训练 loader 统计阶段 | 模式 S 含 TE/端粒/着丝粒；模式 B 无可靠结构注释 | 决定训练 token 在不同功能区域上的目标分布 |

Stage B 的实际抽样逻辑:

1. 先从所有通过 QC、split、防泄漏、区域保留和去冗余的候选窗口中，按 `context_bucket=8K/4K/16K` 分桶。
2. 对每个长度桶内部，再按 4.5 的区域目标比例抽样，例如 CDS、splice、promoter/TSS、UTR、TES、intron、high-quality intergenic、random background。
3. 每抽到一个 8K 窗口，向 Stage B 写入约 8192 token；每抽到一个 4K 窗口，写入约 4096 token；每抽到一个 16K 窗口，写入约 16384 token。
4. 当前方案中 8K 主 context 合格候选全部写入；4K/16K 辅助长度达到受控 replay/warm-up 配额后停止继续写入。
5. 物理 shard 只是存储切分，`shard_000001`、`shard_000002` 等不是新的训练阶段，也不改变长度比例或区域比例。

按长度桶拆分，Stage B 的 `41,242,505,216` token 实际对应:

| 长度桶 | 当前角色 | 实际 token | 等价完整窗口数 |
|---|---|---:|---:|
| 8K | 主 context 全部合格候选 | 32,062,283,776 | 约 3,913,853 个 8K 窗口 |
| 4K | 受控 replay | 6,120,034,304 | 约 1,494,149 个 4K 窗口 |
| 16K | 受控 warm-up | 3,060,187,136 | 约 186,779 个 16K 窗口 |

上表的“等价完整窗口数”按 `实际 token / context length` 估算，用于理解规模；精确窗口数以 `inputs/Stage_B/manifest.tsv` 和 `.windows.tsv.gz` 为准，因为 contig 边界、最后一个 shard、质量过滤、区域候选不足和去冗余会造成不同长度桶内部的轻微偏差。

其他 stage 同理:

| Stage | 主长度逻辑 | replay/warm-up 逻辑 | 实际含义 |
|---|---|---|---|
| Stage B | 8K 占主导 | 4K 保留短 motif/CDS/splice 密度，16K warm-up 让模型提前接触稍长上下文 | 先学局部 DNA 语法和功能位点，不让超长背景过早稀释信号 |
| Stage C1 | 64K 占主导 | 8K、16K/32K、4K replay | 从 Stage B checkpoint 扩到 gene body、promoter-gene、长 intron，同时保持短程功能能力 |
| Stage C2 | 128K 占主导 | 64K 和 8K/16K replay | 进一步学习远端调控和结构上下文，同时避免 8K probe 退化 |
| Stage D | 256K 占主导 | 128K 和少量 8K/64K replay | 资源允许时做长上下文 midtraining，不作为从零训练阶段 |

实际体积说明:

- `input_ids` 使用 `uint8`，约 1 byte/token。
- `.windows.tsv.gz`、manifest、summary、metadata 和 SHA256 是额外开销。
- 当前四个 stage 已全部生成，`training_server_transfer/` 总体约 67GB。
- 第一版不保存固定 mask/label/batch order/RC 结果。

本服务器生成目录:

```text
training_server_transfer/
  README.md
  MANIFEST.tsv
  SHA256SUMS
  configs/
  metadata/
  inputs/
    Stage_B/
      manifest.tsv
      summary.tsv
      shard_000001.input_ids.bin
      shard_000001.windows.tsv.gz
      ...
    Stage_C1/
    Stage_C2/
    Stage_D/
```

训练服务器传输目录保持简约:

```text
training_server_transfer/
  README.md
  MANIFEST.tsv
  SHA256SUMS
  configs/
  metadata/
  inputs/
```

用户实际传输时，只需要把 `training_server_transfer/` 整个目录传到训练服务器。若训练服务器空间紧张，可以只保留当前要训练的 `inputs/Stage_*` 子目录，训练完一个 stage 后删除或归档该 stage 输入。

训练服务器磁盘估算:

- 当前四个 stage 全部搬运，输入和小元数据约 67GB。
- 若每次只保留一个 stage，最大 Stage B 约 39GB。
- 加 checkpoint、cache、logs、临时文件，训练服务器建议至少 800GB 可用空间；若保留多份 checkpoint 或多实验并行，建议 1TB-1.5TB。
- 若使用 8 x 80GB GPU 训练 Large 模型，checkpoint 和 optimizer state 会成为主要额外空间来源，需要严格只保留最近 checkpoint 和 best checkpoint。

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

- 第一版正式训练不再使用 `5%-10%` N 救援。
- CDS、splice、UTR、TES、promoter core 等核心功能窗口默认 `N <= 1%`。
- promoter distal、ordinary gene_body、background 等低价值区域默认 `N <= 0.5%`。
- 64K/128K/256K 长 context 窗口统一要求 `N <= 0.5%`。
- validation/test 使用同等或更严格标准，不做救援。

连续 N:

- CDS、splice、start/stop 监督窗口中连续 `N >= 100 bp` 丢弃。
- promoter distal、ordinary gene_body、background 中连续 `N >= 100 bp` 丢弃。
- 64K/128K/256K 长 context 窗口中连续 `N >= 50 bp` 丢弃。

有效碱基:

- 核心功能窗口要求 `A/C/G/T >= 98%`。
- promoter distal、ordinary gene_body、background 要求 `A/C/G/T >= 99.5%`。

低复杂度:

- 核心功能窗口单一碱基比例 `> 75%` 丢弃。
- promoter distal、ordinary gene_body、background 和长 context 窗口单一碱基比例 `> 72%` 丢弃。
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
| promoter/TSS distal | 约 8%-10% | TSS upstream `5-20 kb` 只保留高质量代表窗口；实现上对 distal promoter 做稳定哈希下采样 |
| weak promoter | 100% of weak window | 无可靠 TSS 时，只用 gene upstream `2 kb` 作为弱 promoter 标签 |
| intron boundary | 100% | exon-intron boundary 两侧 `2 kb` 全部保留 |
| ordinary intron interior | 5%-8% | 普通 intron 内部随机分层保留，优先高质量窗口 |
| long intron interior | 2%-3% | `>20 kb` 长 intron 内部优先保留 GC/复杂度正常、`N <= 0.5%-1%` 的窗口 |
| TE/repeat annotated | 50% | 只有可靠 repeat 注释时启用 |
| gene-proximal TE/repeat | 100% | 距 gene body 或 promoter `20 kb` 内的 TE/repeat 全部保留 |
| TE boundary | 100% | TE 边界上下游 `+/-2 kb` 全部保留 |
| high-confidence telomere | 100% | motif density 达标或有 telomere 注释的端粒窗口全部保留；contig 末端但无 motif 证据不标为 telomere |
| subtelomere | 30%-50% | telomere 内侧高质量窗口，优先保留 TE/gene-rich、低 N、非 gap 片段 |
| centromere core | 100% | 有 CENH3/Hi-C/T2T/satellite 强证据的核心着丝粒窗口全部保留 |
| pericentromere | 20%-40% | centromere 两侧 repeat-rich 区域分层保留，长 context 优先 |
| satellite/tandem repeat high-confidence | 30%-50% | 高置信 satellite/tandem repeat 保留为结构区域；低复杂度但无结构证据的窗口仍按低复杂度过滤 |
| rDNA/organellar insertion | 100% high-confidence; 20% weak | 高置信 rDNA/叶绿体/线粒体核插入全部保留，弱证据仅低权重保留 |
| segmental duplication | 20%-40% | 大段重复/拷贝数相关区域保留代表窗口，避免同一 assembly 内重复过采样 |
| gene-proximal intergenic | 约 5% | 距任意 gene `20 kb` 内，优先 `N <= 0.5%-1%`、非低复杂度、完整覆盖窗口 |
| distal intergenic / far noncoding | 1%-2% | `N <= 0.5%-1%`、无长 N、非低复杂度、非高度重复、GC 在本 genome `5%-95%` 分位范围内 |
| random genome coverage | 0.5%-1% | 从通过 hard filter 的全基因组窗口中额外抽样，避免完全丢失背景分布 |

TE/repeat 规则:

- 有 repeat 注释的 assembly 才启用 TE/repeat 候选池。
- 无 repeat 注释 genome 不把 intergenic 伪标为 non-repeat 或 TE/repeat。
- TE/repeat 只作为区域标签和采样层，不覆盖 CDS、splice、UTR、promoter 等高优先级功能标签。
- telomere/centromere/satellite 只在 high 或 medium 证据下进入候选池；low 证据只作为 quality/structure flag。
- 如果同一窗口同时命中 CDS/splice/promoter 与 TE 或结构区域，主 `region_bucket` 仍按 CDS/splice/promoter 等功能区优先，结构信息写入 `structure_flags`。

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

训练 batch 的目标区域比例分为三种模式。

模式 S: 有可靠 TE/repeat + telomere/centromere/satellite 等结构注释时使用。该模式适合 T2T assembly、已有 repeat annotation 或后续本服务器补齐 EDTA/RepeatMasker/TRF 等注释后的版本。

| 区域 | 采样比例 | loss 权重 | 目的 |
|---|---:|---:|---|
| CDS/protein-coding exon | 22% | 1.50 | 密码子、ORF、保守编码结构 |
| splice donor/acceptor | 14% | 2.00 | 剪接位点和外显子边界 |
| promoter/TSS | 13% | 1.40 | 启动子和表达调控 |
| UTR | 8% | 1.20 | 翻译调控、mRNA 稳定性 |
| TES/polyA | 5% | 1.20 | 转录终止和 polyA |
| intron/gene body | 9% | 0.90 | 长程 gene body 和调控上下文 |
| TE/repeat family | 10% | 0.85 | TE family、repeat grammar、TE 调控背景 |
| TE boundary/intact LTR | 4% | 1.10 | TE 插入边界、完整 LTR、结构变异相关信号 |
| telomere/subtelomere | 4% | 0.90 | 染色体末端重复、subtelomeric 快速演化区域 |
| centromere/pericentromere/satellite | 5% | 0.85 | 着丝粒卫星重复、低重组 repeat-rich 长上下文 |
| rDNA/organellar insertion/segmental duplication | 2% | 0.80 | 特殊重复、核质插入、大段重复 |
| high-quality intergenic | 3% | 0.60 | 高质量非编码背景 |
| random background | 1% | 0.50 | 保留 genome-wide 分布 |

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

当前默认采用模式 B，除非后续为当前 crop assembly 补齐可靠 TE/repeat 注释或外部 repeat annotation。若补齐 telomere/centromere/satellite 等结构注释，则切换到模式 S。

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

采用“按 stage 搬运固化输入”的低磁盘策略:

- 本服务器生成 `training_server_transfer/inputs/Stage_B`、`Stage_C1`、`Stage_C2`、`Stage_D`。
- 每次在 `training_server_transfer/` 中只放当前要训练的 stage 数据、必要索引、配置和 manifest。
- 训练服务器不需要搬运原始 plantDB 全量数据。
- 训练服务器不需要重新解析 FASTA/GFF/GTF。
- 训练服务器读取固化 `input_ids` 或 sequence window。
- 训练时动态生成 mask、MLM labels、RC 增强、batch 顺序和动态 loss。
- 只保留最近 1 个完整 checkpoint 和 1 个 best inference checkpoint。
- 一个 stage 训练完成后，删除或归档该 stage 输入，再搬运下一 stage。

### 6.1 训练服务器磁盘需求

| 项目 | 体积 |
|---|---:|
| 当前 stage 固化输入 | Stage B 最大约 40-120 GB |
| transfer manifest + configs | < 1 GB |
| sequence/annotation/sampling 必要索引 | 10-60 GB |
| logs | 20-50 GB |
| 当前 checkpoint + best model | 100-300 GB |
| results/downstream cache | 20-100 GB |
| 安全余量 | 100-200 GB |

结论:

- 每次只搬一个完整 stage: 最低 500GB 可用磁盘，推荐 800GB。
- 四个 stage 全部同时放训练服务器: 最低 800GB，推荐 1TB-1.5TB。
- 若保存多个 full checkpoint 或 ZeRO optimizer state，推荐 1.5TB-2TB。

### 6.2 训练服务器 CPU/RAM

推荐:

- CPU: 32-64 核。
- RAM: 128-256GB。
- 本地 SSD 优先放当前 stage 输入、checkpoint 和 dataloader 临时 cache。

最低:

- CPU: 24-32 核。
- RAM: 96-128GB。

## 7. 模型输入

每个训练样本由固化输入和训练时动态监督两部分组成。

本服务器固化:

```text
input_ids:       [L] uint8, 或 sequence window
region_ids:      [L] uint8
structure_flags: [L] uint16 optional, telomere/centromere/satellite/TE/rDNA/duplication 等多标签位标记
repeat_family_id:[L] int32 optional, TE/repeat family；未知或未启用为 -1
region_weights:  [L] fp16/bf16 base value
quality_scores:  [L] optional fp16 or compact flags
metadata:        assembly_id, species_id, genus_id, contig_id, start, end, strand, split, context_bucket
```

训练服务器动态生成:

```text
input_ids:       [B, L] int64
labels_mlm:      [B, L] int64, 非 mask 位点为 -100
loss_mask:       [B, L] bool
region_ids:      [B, L] uint8, 来自固化输入
structure_flags: [B, L] uint16 optional
repeat_family_id:[B, L] int32 optional
region_weights:  [B, L] fp16/bf16, 可在 base value 上动态调整
quality_scores:  [B, L] optional fp16, 来自固化输入或 flags
rc_flag:         [B] bool
metadata:        assembly_id, species_id, genus_id, contig_id, start, end, strand, split
```

关键原则:

- 固化输入不包含固定 mask。
- 固化输入不包含固定 `labels_mlm`。
- 同一个 `input_ids` 在不同 epoch 可产生不同 mask 和 RC 增强。
- stage 输入顺序不作为训练 batch 顺序，训练服务器 dataloader 仍可 shuffle。

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

结构标签编码:

| 字段 | 含义 |
|---|---|
| `structure_flags` | 位标记，不互斥；可同时表示 `TE`, `TE_boundary`, `telomere`, `subtelomere`, `centromere`, `pericentromere`, `satellite`, `tandem_repeat`, `rDNA`, `organellar_insertion`, `segmental_duplication`, `synteny_breakpoint` |
| `repeat_family_id` | TE/repeat family 编号；仅在 RepeatMasker/EDTA/RepeatModeler 或外部 GFF 能给出 family/class 时启用 |
| `structure_confidence` | optional，high/medium/low 编码；low 不用于监督正例，只用于过滤和分析 |

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
- structure flag embedding 和 repeat family embedding，只有结构注释可靠时启用。
- optional strand/quality embedding。
- RC-equivariant bidirectional Mamba2/Hyena backbone。
- v1.1 可升级为 CrossDNA 启发的显式动态 cross-strand 双分支结构，在 forward 与 reverse-complement 分支之间加入轻量链间通信。
- 每 4-6 层插入 local/global sparse attention。
- RMSNorm。
- SwiGLU/gated MLP。
- MLM head。
- causal auxiliary head。
- optional structure multi-label head: telomere、subtelomere、centromere、pericentromere、satellite、TE boundary、rDNA、segmental duplication 等。
- optional TE/repeat family head: 只在 family-level repeat annotation 可靠时启用。
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
  + 0.03 * L_structure_multilabel_optional
  + 0.02 * L_repeat_family_optional
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

### 9.3.1 CrossDNA 启发的显式 cross-strand objective

文献 `Explicit dynamic cross-strand interactions for DNA sequence language modelling` 的核心启发是: DNA language model 不应只依赖 RC augmentation 或静态 RC consistency，而应让 forward strand 与 reverse-complement strand 在模型内部显式、动态交互。CrossDNA 的 duplex-inspired dual-branch 思路可作为 CropGenome-FM v1.1 的升级方向。

本项目采用保守升级:

1. 不停止当前 `v1-backbone` Stage B。当前训练继续作为正式 backbone checkpoint。
2. Stage B 达到可用 checkpoint 后，新增 `v1.1-cross-strand-midtraining`:
   - 输入同一窗口的 `x_forward` 和 `x_reverse_complement`。
   - 两个分支共享或部分共享 embedding/backbone 参数。
   - 每 4-6 层加入一次轻量 `CrossStrandCommunication`。
   - 将 RC 分支 hidden state reverse 回 forward 坐标后做 gated fusion。
3. 训练目标在原有 loss 上加入:

```text
L_cross_strand =
  L_MLM_forward
  + L_MLM_reverse_complement
  + 0.03 * L_cross_strand_embedding_alignment
  + 0.02 * L_cross_strand_logit_alignment
```

4. 若显存紧张，cross-strand midtraining 先使用:
   - 8K context。
   - micro_batch_size 减半。
   - cross-strand communication 只在少数层启用。
   - 不在 64K/128K 阶段立即全量启用。
5. 只有当 v1.1 在 RC consistency、splice、promoter/TSS、enhancer/open chromatin、variant effect 等 probe 上优于 v1-backbone，才把 cross-strand block 纳入 C1/C2 长上下文训练。

预期收益:

- 降低方向偏置，比单纯 RC augmentation 更稳定。
- 对 motif orientation、双链互补相关的调控模式、剪接边界和变异效应评分更有利。
- 对 enhancer/promoter 这类文章中强调的调控序列任务，预计比当前隐式 RC consistency 版本更强。

风险:

- 显式双分支会增加约 1.5-2 倍显存和计算量。
- 若 cross-strand communication 太频繁，8K/64K 训练吞吐会明显下降。
- 作物长基因组和 repeat-rich 区域比 human 2K benchmark 更复杂，必须用本项目的跨属/跨 assembly holdout probe 验证，不能直接假设收益。

### 9.4 Structure and repeat auxiliary losses

结构辅助 loss 只在 high/medium 证据标签存在时启用，且权重低于主 MLM，避免模型为学习重复分类而牺牲 CDS/splice/TSS 等核心功能。

- `L_structure_multilabel_optional`: 多标签 BCE/focal loss，标签包括 telomere、subtelomere、centromere、pericentromere、satellite/tandem repeat、TE boundary、rDNA、organellar insertion、segmental duplication。
- `L_repeat_family_optional`: TE/repeat family classification，只有 EDTA/RepeatMasker/RepeatModeler 或外部 repeat GFF 能提供 family/class 时启用。
- low confidence 结构标签不作为正例监督，只用于分组 loss 统计和错误分析。
- 对 telomere/centromere/satellite 这类高度重复区域，MLM mask 使用较长 span 和较低 loss 权重，防止模型过度记忆简单重复。

## 10. 训练阶段、资源和时间

### 10.1 CPU 数据处理阶段

| 阶段 | 内容 | 资源 | 时间 |
|---|---|---|---:|
| P1 | assembly manifest | 8-16 核，32GB | < 2 小时 |
| P2 | FASTA QC 扫描 | 2-6 作业，每作业 30 核，100-150GB | 8-24 小时 |
| P3 | GFF/GTF 解析 | 2-6 作业，每作业 30 核，80-150GB | 8-36 小时 |
| P4 | 区域构建和候选池 | 2-6 作业，每作业 30 核，80-150GB | 6-24 小时 |
| P5 | split 和防泄漏检查 | 8-16 核，32-64GB | 2-8 小时 |
| P6 | stage 输入固化和 shard 校验 | 2-6 作业，每作业 30 核，80-150GB | 8-36 小时 |
| P7 | `training_server_transfer/` 目录准备和传输校验 | 8-16 核，32-64GB | 1-6 小时 |

CPU 总体: 3-7 天，取决于是否一次性生成全部 Stage B/C1/C2/D 输入。

### 10.2 GPU 预训练阶段

正式采用“同一模型渐进式扩长 + 每阶段短长度 replay”的训练策略。不是 8K、64K、128K、256K 四个模型分别从零训练，也不是把所有长度从一开始等比例混在一起训练。

核心原则:

- Stage B 先训练 8K，建立局部 DNA 语法、CDS、splice、TSS/TES 表征。
- Stage C1 从 Stage B checkpoint 继续训练到 64K。
- Stage C2 从 Stage C1 checkpoint 继续训练到 128K。
- Stage D 从 Stage C2 checkpoint 继续做 256K midtraining，资源允许后执行。
- 每个阶段设置一个主 context length；当前固化输入采用“主 context 合格候选全部写入 + 辅助 context 受控 replay/warm-up”。
- 长度组成以 `summary.tsv` 和 `manifest.tsv` 实际 token 为准，不再使用旧版固定 token 预算截断主 context。

| 阶段 | 主 context | 长度组成逻辑 | 已固化 token | 推荐 GPU | 预计时间 | 目标 |
|---|---:|---|---:|---:|---:|---|
| Stage B | 8K | 全部 8K 主候选 + 4K/16K 受控 warm-up/replay | 41.24B | 4-8 x 80GB | 5-18 天 | 局部 motif、CDS、splice、TSS/TES、TE boundary 初步表示 |
| Stage C1 | 64K | 全部 64K 主候选 + 4K/8K/16K/32K 受控 replay | 20.47B | 8 x 80GB | 7-18 天 | gene body、promoter-gene、长 intron、subtelomere/pericentromere |
| Stage C2 | 128K | 全部 128K 主候选 + 8K/16K/64K 受控 replay | 6.90B | 8 x 80GB | 5-14 天 | 远端调控、TE-rich region、centromere/subtelomere 上下文 |
| Stage D | 256K | 全部 256K 主候选 + 8K/64K/128K 受控 replay | 2.78B | 8-16 x 80GB | 4-14 天 | 资源允许后做染色体结构长上下文 midtraining |

不采用完全多长度混合的原因:

- 早期直接混入大量 64K/128K/256K 会显著降低吞吐，拉高显存和通信压力。
- 模型尚未学会局部 DNA 语法时，超长窗口收益低，反而容易被 intron/intergenic 背景稀释 CDS/splice/start/stop 信号。
- 不同长度在同一 micro-batch 内混放会增加 padding 和动态 shape 开销，训练统计也更难解释。

### 10.3 多长度 stage input loader 和 batch 组织

本服务器生成 stage 输入时使用两级抽样:

1. 先抽 `context_bucket`: `4K`, `8K`, `16K`, `32K`, `64K`, `128K`, `256K`。
2. 再在该长度桶内按区域比例抽 `region_bucket`: CDS、splice、TSS、UTR、TES、intron、TE/repeat、telomere/subtelomere、centromere/pericentromere、satellite、intergenic、background。

执行口径: 当前版本不再用旧版 `70% 8K` 作为 Stage B 主 context 截断预算。Stage B 的 8K、Stage C1 的 64K、Stage C2 的 128K、Stage D 的 256K 都是该 stage 的主 context；主 context 中通过质量过滤、区域规则、去冗余和 split 防泄漏的候选全部写入。4K/8K/16K/32K/64K/128K 等非主长度只作为受控 replay 或 warm-up，防止短程功能 probe 退化，并在 `summary.tsv` 中记录实际 token 偏差。

训练服务器 dataloader 读取已经固化的 stage 输入，并在训练时组织 batch:

- 同一个 micro-batch 内使用同一个 context length，减少 padding 和显存波动。
- 不同 context length 通过 gradient accumulation 组合到同一个 optimizer step。
- 每个 optimizer step 记录实际 token 数、context_bucket 比例、region_bucket 比例。
- 每个长度桶在本服务器固化前必须已经执行 4.2 硬质量过滤、4.3 候选池保留比例、4.4 去冗余控制、4.5 区域采样比例。
- 长 context 阶段不允许被 intergenic 背景吞掉；CDS/splice/start/stop replay 必须保留。
- 训练服务器只负责 shuffle、dynamic mask、dynamic labels、RC augmentation、batch collation，不重新做全基因组采样。

每阶段日志至少记录:

- `loss_4k`, `loss_8k`, `loss_16k`, `loss_32k`, `loss_64k`, `loss_128k`, `loss_256k`，未启用的桶记为 `NA`。
- 每个 context_bucket 的 tokens/s、GPU memory、mask token 数、region token 分布。
- 每个阶段结束时保存当前主 context checkpoint、最优 8K probe checkpoint 和最优 long-context checkpoint。

### 10.4 阶段评估门槛

进入下一阶段前必须通过以下检查:

| 检查项 | 目的 | 通过标准 |
|---|---|---|
| `val_loss_8k` | 短程语法是否保留 | 相比上一阶段不能明显变差 |
| 当前主 context `val_loss` | 当前长度是否有效学习 | 持续下降或达到稳定平台期 |
| splice donor/acceptor probe | 核心短程功能任务 | C1/C2 后 AUROC/AUPRC 下降不超过 1%-2% |
| CDS frame/start-stop probe | 编码区语法 | 不明显低于 Stage B checkpoint |
| TSS/TES probe | 中程调控能力 | C1 后应优于 Stage B 或至少不退化 |
| long intron/promoter-gene probe | 长程上下文能力 | C1/C2 应较 Stage B 有提升 |
| TE boundary / repeat family probe | 重复序列和插入边界表示 | 启用 TE 注释后应优于随机初始化和 DNABERT-2 embedding |
| telomere/centromere/satellite probe | 染色体结构区域表示 | 只在 high-confidence 标签上评估；C2/D 应优于 Stage B |
| RC consistency | 双链一致性 | 长阶段不能退化 |
| tokens/s 和 GPU memory | 工程可训练性 | 达到可持续训练吞吐，无频繁 OOM |

如果 C1/C2 的长 context loss 下降，但 8K 下游 probe 明显退化，需要提高短 context replay 比例；如果短程任务稳定但 long-context probe 没有提升，需要增加 64K/128K 主长度 token 比例或延长当前阶段。

### 10.5 推荐训练服务器和总体时间

推荐训练服务器:

- 磁盘: 每次只搬一个 stage 时最低 500GB 可用空间，推荐 800GB；若多个 stage 同时保存，推荐 1TB-1.5TB。
- CPU: 32-64 核。
- RAM: 128-256GB。
- GPU: 推荐 4-8 张 80GB；最低 2 张 80GB 可跑 Base 或 Large 8K/64K。

总体时间:

- CPU 预处理和 stage 输入固化: 3-7 天。
- 8K + 64K/128K 渐进式正式训练: 4-8 周。
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
| TE family classification | family-level TE/repeat annotation | other repeat families + matched non-repeat | assembly/species holdout | macro-F1, AUROC | 结构区域预训练和 repeat family embedding 应优于普通 DNA embedding |
| TE insertion boundary | TE 起止边界 +/-2kb | TE 内部、matched intergenic | family + species holdout | AUROC, AUPRC, boundary F1 | TE boundary 专门采样，预计优于未显式建模 TE 的基线 |
| intact LTR / solo LTR | intact LTR 注释 | fragmented/solo/other repeat | assembly holdout | AUROC, macro-F1 | 长上下文可利用 LTR pair 和内部结构 |
| telomere/subtelomere detection | high-confidence telomere/subtelomere | contig edge non-telomere + distal intergenic | assembly holdout；低置信不进测试 | AUROC, AUPRC | motif + 长上下文结合，预计优于只看 k-mer density 的规则模型 |
| centromere/pericentromere detection | CENH3/Hi-C/T2T/satellite 支持区域 | matched repeat-rich non-centromere | species/assembly holdout | AUROC, AUPRC | 通过 satellite、TE 和长程重复上下文，预计优于短窗口模型 |
| satellite/tandem repeat class | high-confidence tandem/satellite repeat | low-complexity non-satellite + random repeat | assembly holdout | macro-F1 | 区分结构性卫星重复和低复杂度垃圾窗口 |
| rDNA / organellar insertion | 45S/5S rDNA、cp/mt nuclear insertion | matched repeat/intergenic | assembly holdout | AUROC, AUPRC | 减少特殊重复被误学为普通 intergenic |
| segmental duplication / CNV-prone region | self-alignment 或已有 duplication 注释 | unique matched regions | assembly/species holdout | AUROC, AUPRC | 对大段重复和变异热点有更好表征 |

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
6. TE insertion boundary 和 TE family classification，前提是 repeat annotation 可靠。
7. telomere/centromere/satellite probe，前提是 high-confidence 结构标签足够。

不保证全面超过的任务:

- expression regression。
- chromatin/open region 跨组织泛化。
- 农艺变异 effect size 排序。
- telomere/centromere 跨物种泛化；这类区域高度物种特异，必须报告按物种/属分层结果。

所有“优于基线”的说法必须以后续严格 benchmark 为准。

### 12.1 结构基因组增强的文献和工具依据

本计划纳入 telomere、centromere、TE/repeat、satellite/tandem repeat 等信息的原因:

- AgroNT 已证明植物基因组 foundation model 可用于 regulatory annotation、promoter/terminator、gene expression 和 variant prioritization 等任务；本项目在此基础上增加结构注释感知和更长 context。
- EDTA/RepeatMasker/RepeatModeler 是植物 TE/repeat 注释的主要候选工具；其中 EDTA 面向 de novo TE annotation 和高质量非冗余 TE library 构建。
- Tandem Repeats Finder、TRASH、RepeatOBserver 等工具可用于 tandem repeat/satellite repeat 探测，适合支持 telomere、centromere 和 satellite 弱标签。
- T2T 植物基因组研究显示，完整 assembly 能解析普通参考基因组常缺失的 telomere、centromere、satellite arrays、rDNA cluster 和 repeat-rich 结构区域；这类区域应作为长上下文模型的关键补充。

可参考资料:

- AgroNT edible plant genome foundation model: https://www.nature.com/articles/s42003-024-06465-2
- EDTA TE annotation: https://github.com/oushujun/EDTA
- TRASH tandem repeat annotation: https://pmc.ncbi.nlm.nih.gov/articles/PMC10199239/
- Tandem Repeats Finder: https://tandem.bu.edu/trf/trf.html
- Plant centromere structure/evolution review: https://genome.cshlp.org/content/34/2/161
- T2T wheat genome example: https://www.nature.com/articles/s41588-025-02137-x

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
    te_family/
    te_boundary/
    intact_ltr/
    telomere_subtelomere/
    centromere_pericentromere/
    satellite_tandem_repeat/
    rdna_organellar_insertion/
    segmental_duplication/
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
- 按区域分组 loss: CDS、splice、promoter/TSS、TES、UTR、intron、TE/repeat、telomere/subtelomere、centromere/pericentromere、satellite、intergenic。
- 按属/物种分组 val loss。
- RC consistency 指标。
- tokens/s 和 GPU 显存峰值。
- stage input loader 吞吐。
- dynamic mask/label 生成开销。
- 当前 `training_server_transfer/` 目录版本和 stage input manifest。

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
| 待填 | TE family classification | assembly/species holdout | CropGenome-FM | 待填 | 待填 | 待填 | macro-F1 待填 | 待填 | 待填 | 待填 | 待填 |
| 待填 | TE insertion boundary | family + species holdout | CropGenome-FM | 待填 | 待填 | 待填 | boundary F1 待填 | 待填 | 待填 | 待填 | 待填 |
| 待填 | intact LTR / solo LTR | assembly holdout | CropGenome-FM | 待填 | 待填 | 待填 | macro-F1 待填 | 待填 | 待填 | 待填 | 待填 |
| 待填 | telomere/subtelomere | assembly holdout | CropGenome-FM | 待填 | 待填 | 待填 | high-confidence only | 待填 | 待填 | 待填 | 待填 |
| 待填 | centromere/pericentromere | species/assembly holdout | CropGenome-FM | 待填 | 待填 | 待填 | high-confidence only | 待填 | 待填 | 待填 | 待填 |
| 待填 | satellite/tandem repeat | assembly holdout | CropGenome-FM | 待填 | 待填 | 待填 | macro-F1 待填 | 待填 | 待填 | 待填 | 待填 |
| 待填 | rDNA/organellar insertion | assembly holdout | CropGenome-FM | 待填 | 待填 | 待填 | - | 待填 | 待填 | 待填 | 待填 |
| 待填 | segmental duplication | assembly/species holdout | CropGenome-FM | 待填 | 待填 | 待填 | - | 待填 | 待填 | 待填 | 待填 |

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
6. 在本服务器按 Stage B/C1/C2/D 生成固化输入窗口或 `input_ids`，并为每个 stage 生成 shard manifest 和 sha256 校验。
7. 准备 `training_server_transfer/`，每次只放当前要搬运的 stage、必要索引、配置和 manifest。
8. 将 `training_server_transfer/` 整体传输到训练服务器，并在训练服务器校验 sha256、manifest、stage mix 比例和 split。
9. 在训练服务器跑 1000 step dry-run，校正 dataloader 吞吐、dynamic mask 开销、显存和 checkpoint 策略。
10. 启动 Stage B 8K 正式预训练。
11. 完成 Stage B 后做第一批下游 probe，删除或归档 Stage B 输入，准备 Stage C1 transfer。
12. 按 Stage C1 -> C2 -> D 顺序继续预训练。

## 15. 进展记录

- 2026-06-07 23:26:04 CST: 读取 `/home/user/zhangzhishuai/data/plantDB/genome/README.md`，确认训练数据口径；完成 AgroNT、PlantCAD/Caduceus、Evo 2、HyenaDNA、DNABERT-2、GROVER 和 DNA foundation benchmark 调研；确定主路线为长上下文、单碱基、RC 等变双向 Mamba/Hyena 模型。
- 2026-06-07 23:46:31 CST: 按用户要求扩展为端到端正式训练方案，补充 `.fna.gz` 扫描、contig QC、split、窗口化、token shard、GPU batch 输入、mask/采样策略、下游监督数据构建，以及 CPU/GPU 资源和每阶段耗时估算。
- 2026-06-08 11:42:31 CST: 按用户要求重构方案，放弃无结构注释 genome，正式数据限定为有 FASTA+GFF/GTF 的 crop assembly；新增区域加权采样、严格防泄漏 split、片段过滤、跨服务器搬运磁盘估算、详细下游任务和基线优势预期。
- 2026-06-08 15:00:51 CST: 根据用户确认，放弃进一步压缩到核心 assembly 的方案，整理当时版本的完整训练计划；后续已按 22:19:54 CST 的跨服务器固化输入方案更新。
- 2026-06-08 18:16:18 CST: 按用户要求新增评测结果记录规范，预置预训练指标表、下游任务结果表、基线比较表和结果解释规则；明确 GitHub 只记录摘要和关键表格，不上传大结果文件。
- 2026-06-08 18:45:35 CST: 按用户要求参考 `douke_genome` 的区域采样方案，调整为“有 TE/repeat 注释模式”和“无可靠 TE/repeat 注释 fallback 模式”；当前默认使用无 TE fallback，避免把普通 intergenic 伪标为 repeat/non-repeat。
- 2026-06-08 19:15:22 CST: 按用户给定的 4.5.1-4.5.3 规则重构输入处理: 增加硬质量过滤、候选池区域保留比例、minimizer/simhash 去冗余和 assembly/genus 级 intergenic token 上限；明确候选池保留比例不同于训练 batch 采样比例。
- 2026-06-08 21:06:31 CST: 根据临时 context 长度策略评估，正式将 GPU 预训练优化为“同一模型渐进式扩长 + 每阶段短长度 replay”；新增多长度 sampler、micro-batch 组织、context_bucket 日志和阶段进入门槛。
- 2026-06-08 22:19:54 CST: 按用户当前确认的跨服务器训练思路重写训练计划: 本服务器负责索引、候选池和 Stage B/C1/C2/D 固化输入生成；训练服务器只接收 `training_server_transfer/` 目录，训练时动态生成 mask、label、RC 增强和 batch 顺序。
- 2026-06-08 22:43:02 CST: 开始 CPU 数据处理。由于 `/home/user/zhangzhishuai/data/plantDB/genome` 持续下载且可能包含非作物，manifest 生成加入作物属白名单；今晚实际 crop manifest 为 263 个 assembly、26 个属。已提交 Slurm `cu` 分区任务: FASTA QC array `8468483`、annotation QC array `8468489`、依赖合并任务 `8468495`；每个 array task 4 核 24GB，`--time=09:10:00`，脚本内设置 2026-06-09 08:00:00 CST 截止。
- 2026-06-09 08:48:28 CST: 按用户要求后续任务切换到 `q07/q08`。FASTA QC 263/263 成功，annotation QC 经流式脚本重跑后 263/263 成功；已在 `q07` 完成合并，得到 `sequence_index/contigs.tsv`、`annotation_index/features.tsv` 等文件；已在 `q08` 提交 region/sampling candidate 构建任务 `8469374`。
- 2026-06-09 10:28:40 CST: q08 region/sampling candidate 构建任务 `8469374` 完成，生成 `sampling_index/region_candidates.tsv.gz`；q07 split/transfer 任务完成，生成第一版基础传输包并完成 SHA256 校验。
- 2026-06-09 21:10:00 CST: Stage B/C1/C2/D 固化输入全部完成，`training_server_transfer/` 整理为 `configs/metadata/inputs` 简约目录，总体约 50GB；全目录 `sha256sum -c SHA256SUMS` 通过。
