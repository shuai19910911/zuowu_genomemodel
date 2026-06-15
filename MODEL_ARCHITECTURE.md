# CropGenome-FM 模型结构解析

更新时间: 2026-06-08 22:19:54 CST

## 1. 设计结论

CropGenome-FM 是一个面向作物结构注释基因组的区域加权长上下文基础模型。第一版正式模型只使用 262 个有 FASTA + GFF3/GTF 的 assembly，不使用缺少结构注释的 genome。训练数据在本服务器按 stage 固化输入窗口或 `input_ids`，训练服务器动态生成 mask、label、RC 增强和 batch 顺序。

主架构:

```text
single-base DNA tokens
  + region / strand / quality embeddings
  + RC-equivariant bidirectional Mamba2-Hyena backbone
  + periodic local/global attention
  + region-weighted MLM head
  + causal likelihood auxiliary head
  + RC consistency objective
```

## 2. 为什么这样设计

| 需求 | 设计 |
|---|---|
| SNP/indel 变异评分 | 单碱基 token，避免 k-mer/BPE 边界改变 |
| 作物结构注释任务 | 用 GFF/GTF 构建 CDS、UTR、splice、promoter、TES、intron、intergenic 区域标签 |
| 长程调控和 gene body | 8K -> 64K -> 128K -> 256K 课程训练 |
| DNA 双链一致性 | reverse-complement equivariant block 和 RC consistency loss |
| 前沿长上下文效率 | Mamba2/Hyena 为主体，periodic attention 捕获精确局部/中程交互 |
| 下游迁移 | 输出 per-base likelihood、sequence embedding、region-aware embedding |

## 3. 输入定义

### 3.1 token vocabulary

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

IUPAC ambiguous bases 默认转为 N。N/PAD 不参与主 loss。

### 3.2 模型 batch

```text
input_ids:       [B, L] int64
labels_mlm:      [B, L] int64, non-mask = -100
loss_mask:       [B, L] bool
region_ids:      [B, L] uint8
region_weights:  [B, L] fp16/bf16
quality_scores:  [B, L] optional fp16
rc_flag:         [B] bool
metadata:        assembly_id, species_id, genus_id, contig_id, start, end, strand
```

上下文长度:

- Stage B: 8192。
- Stage C1: 65536。
- Stage C2: 131072。
- Stage D: 262144；512K 作为后续扩展。

## 4. 区域 embedding 和权重

区域 id:

- CDS/exon。
- splice donor/acceptor flank。
- UTR。
- promoter/TSS。
- terminator/TES/polyA flank。
- intron。
- TE/repeat annotated，可选，只有可靠注释时启用。
- high-quality intergenic。
- random background。

模型输入来自候选池，而不是全基因组无差别切片。候选池先经过硬质量过滤:

- train 默认 `N <= 5%`，validation/test 必须 `N <= 5%`；`5%-10%` 只允许稀缺小属或稀缺功能区低权重救援。
- 任意连续 `N >= 1 kb` 丢弃；CDS/splice/start/stop 监督窗口中连续 `N >= 100 bp` 丢弃。
- 普通窗口 `A/C/G/T >= 90%`；关键监督窗口 `A/C/G/T >= 98%`。
- 单一碱基比例 `> 80%` 或 dust/entropy 低复杂度的纯背景窗口丢弃。
- contig/scaffold 两端不足 `1 kb` 的窗口默认丢弃，除非包含完整基因结构。
- CDS 坐标越界、transcript parent 缺失、CDS 长度不成 3 倍数的区域不用于 CDS/frame/splice 监督。

候选池保留比例:

| 区域 | 候选池保留比例 |
|---|---:|
| CDS / coding exon | 100% |
| splice donor/acceptor +/-2 kb | 100% |
| start/stop codon +/-2 kb | 100% |
| annotated UTR and transcript boundary | 100% |
| TSS upstream 0-5 kb | 100% |
| TSS upstream 5-20 kb | 15% |
| exon-intron boundary +/-2 kb | 100% |
| ordinary intron interior | 10% |
| long intron interior >20 kb | 5% |
| TE/repeat annotated interval | 50% |
| TE/repeat within 20 kb of gene/promoter | 100% |
| TE boundary +/-2 kb | 100% |
| gene-proximal intergenic within 20 kb | 10% |
| distal intergenic / far noncoding | 3%-5% |
| random genome coverage | 1%-2% |

去冗余控制:

- 普通 intergenic 和 repeat-rich 背景相似度 `>= 95%` 只保留 1 个代表。
- CDS、splice、start/stop 不因相似性丢弃，只做质量过滤。
- 每个 assembly 的 distal intergenic token 占比不超过该 assembly 训练 token 的 `5%`。
- 每个属的 ordinary intergenic token 占比不超过该属训练 token 的 `10%`。

批内采样比例和 loss 权重参考 `douke_genome`，但当前作物数据若无可靠 TE/repeat 注释，则使用 fallback，不伪造 repeat 标签。候选池保留比例用于控制磁盘和冗余，下面的采样比例用于控制每个训练 batch 的学习重点。

模式 A: 有可靠 TE/repeat 注释。

| 区域 | 采样比例 | loss 权重 |
|---|---:|---:|
| CDS/exon | 25% | 1.50 |
| splice donor/acceptor | 15% | 2.00 |
| promoter/TSS | 15% | 1.40 |
| UTR | 10% | 1.20 |
| TES/polyA | 5% | 1.20 |
| intron | 10% | 0.90 |
| TE/repeat annotated | 12% | 0.80 |
| high-quality intergenic | 5% | 0.60 |
| background | 3% | 0.50 |

模式 B: 当前默认，无可靠 TE/repeat 注释。

| 区域 | 采样比例 | loss 权重 |
|---|---:|---:|
| CDS/exon | 28% | 1.50 |
| splice donor/acceptor | 18% | 2.00 |
| promoter/TSS | 15% | 1.40 |
| UTR | 10% | 1.20 |
| TES/polyA | 7% | 1.20 |
| intron | 12% | 0.90 |
| high-quality intergenic | 7% | 0.60 |
| background | 3% | 0.50 |

## 5. Backbone

### 5.1 前沿架构对比

| 方向 | 代表 | 优点 | 局限 | 本项目取舍 |
|---|---|---|---|---|
| 6-mer Transformer MLM | AgroNT, Nucleotide Transformer | 植物任务强基线，训练稳定 | 短上下文，变异边界不自然 | 基线 |
| BPE DNA Transformer | DNABERT-2, GROVER | token 少，效率高 | 单点变异解释受 tokenization 影响 | 基线/效率对照 |
| Hyena long DNA | HyenaDNA, Evo 2 | 单碱基、长上下文、likelihood 评分 | 工程复杂 | 主体方向 |
| RC Mamba/Caduceus | Caduceus, PlantCAD | DNA 双链归纳偏置强 | 需要适配区域加权和长上下文 | 主体方向 |
| Transformer-Mamba2 hybrid | HybriDNA | 注意力 + SSM 互补 | 显存和实现复杂 | periodic attention |
| Explicit cross-strand DNA LM | CrossDNA | 显式双分支建模 forward 与 reverse-complement，并通过轻量 cross-strand communication 做动态双链信息交换；方向鲁棒性和 enhancer 等调控任务表现更强 | 当前公开实现主要是 2K/human reference 预训练和小参数模型；需适配作物长上下文、区域加权和结构注释 | v1.1 升级方向 |
| 监督长调控模型 | AlphaGenome, Enformer, Borzoi | 调控预测强 | 依赖大规模标签，不是纯预训练 | 下游头和评测参考 |

### 5.2 推荐 block

每个 block:

1. RMSNorm。
2. RC-equivariant bidirectional Mamba2 或 Hyena mixer。
3. 每 4-6 层插入 local/global sparse attention。
4. SwiGLU/gated MLP。
5. residual connection。
6. dropout/drop-path 只在过拟合时启用。

推荐 Large 配置:

| 项 | 值 |
|---|---:|
| layers | 32 |
| hidden | 1024 |
| MLP ratio | 4 |
| attention interval | every 4 or 6 layers |
| params | 300M-450M |
| precision | bf16 |
| optimizer | AdamW |
| context | 8K -> 64K -> 128K -> 256K |

### 5.3 CrossDNA 启发的 v1.1 双链交互升级

2026 年 Nature Machine Intelligence 文章 `Explicit dynamic cross-strand interactions for DNA sequence language modelling` 提出 CrossDNA，其关键区别是把 DNA 双链关系从“数据增强或 RC 等变约束”升级为“显式、动态的跨链交互”。文章和官方代码显示，CrossDNA 使用 duplex-inspired dual-branch 架构，同时处理 forward 与 reverse-complement 视图，并通过 lightweight cross-strand communication module 建立链间通信；同时结合 recurrent long-context backbone 和 sliding-window attention。

对 CropGenome-FM 的启发:

1. 当前 v1-backbone 的 RC augmentation 和 `L_RC_consistency` 仍属于隐式双链建模，只约束方向一致，不显式让 forward/RC 两个视图交换信息。
2. v1.1 应在 Stage B checkpoint 之后引入 `CrossStrandBlock`，而不是中断当前训练重来。这样可保留已学到的作物局部语法，再做 cross-strand continued pretraining。
3. 推荐 block 形式:

```text
forward ids x_f
reverse-complement ids x_rc

h_f  = shared_or_tied_backbone_block(x_f)
h_rc = shared_or_tied_backbone_block(x_rc)

c_f, c_rc = CrossStrandCommunication(h_f, h_rc)
h_f  = h_f  + gated(c_f)
h_rc = h_rc + gated(c_rc)

fused = orientation_invariant_pool(h_f, reverse_back(h_rc))
```

4. `CrossStrandCommunication` 第一版只用轻量实现: low-rank cross attention、gated token-wise fusion 或每 N 层一次的 local cross-attention；不在每层全量 cross-attention，避免 8K/64K 长上下文显存爆炸。
5. 作物场景中，显式双链交互优先用于 enhancer/promoter、splice、TSS/TES、motif orientation、variant effect 和 strand-biased annotation 任务；结构区域如 TE boundary、telomere、centromere-like 也可评估方向鲁棒性。
6. 新增评测指标: `cross_strand_delta = |score(x) - score(RC(x))|`，以及 forward/RC embedding cosine、variant effect RC consistency、enhancer/promoter orientation robustness。

实施顺序:

- v1: 当前已启动的 `v1-backbone` Stage_B 继续训练，不中断。
- v1.1: 从 Stage_B checkpoint 做 8K cross-strand midtraining，先只打开每 4-6 层一次的轻量 cross-strand communication。
- v1.2: 如果 v1.1 在 RC consistency、enhancer/promoter、splice 和 variant effect probe 上优于 v1，再把 cross-strand block 带入 C1/C2 长上下文阶段。

## 6. 输出

训练输出:

```text
logits_mlm:          [B, L, vocab]
logits_causal:       [B, L, vocab] optional
hidden_states:       [B, L, H]
sequence_embedding:  [B, H]
region_embedding:    [B, R, H] optional pooled by region
```

推理输出:

- 每碱基 likelihood。
- ref/alt delta likelihood。
- 序列 embedding。
- 区域 embedding。
- 下游任务 logits/regression value。

## 7. Loss

```text
L_total =
  1.00 * L_region_weighted_MLM
  + 0.10 * L_causal_next_token
  + 0.05 * L_RC_consistency
  + 0.05 * L_region_contrastive_optional
```

### 7.1 Region-weighted MLM

- mask rate: 15%。
- span length: 1-512 bp。
- CDS/splice/TSS 区域增加短 span 比例，保证位点级监督。
- intron/intergenic 混合长 span，学习长程上下文。
- N/PAD/低质量位点不计 loss。

### 7.2 Causal auxiliary loss

目的不是把模型变成纯生成模型，而是提供 ref/alt likelihood score。只在部分 batch 或长上下文阶段启用，权重较低。

### 7.3 RC consistency

同一窗口 forward 和 reverse-complement 的 pooled embedding、masked logits 或 variant score 应一致。这个 loss 用于降低 DNA 方向偏置。

## 8. 训练资源估算

| 阶段 | context | 推荐 GPU | token 预算 | 预计时间 |
|---|---:|---:|---:|---:|
| Stage B | 8K | 4-8 x 80GB | 30B-80B | 5-18 天 |
| Stage C1 | 64K | 8 x 80GB | 15B-40B | 7-18 天 |
| Stage C2 | 128K | 8 x 80GB | 5B-20B | 5-14 天 |
| Stage D | 256K | 8-16 x 80GB | 2B-10B | 4-14 天 |

如果只有 2 张 80GB GPU，建议正式模型停在 64K/128K，不强推 256K。

## 9. 评估和基线优势预期

最可能优于基线的任务:

- splice donor/acceptor: splice 区域高采样 + 高 loss 权重。
- TIS/TTS: CDS 高采样 + 单碱基建模。
- CDS/UTR/intron 区域分类: 直接来自区域感知预训练。
- promoter/TES 分类: 长上下文 + TSS/TES 区域采样。

相对基线优势:

- 对 DNABERT-2: 单碱基变异解释和长上下文更自然。
- 对 AgroNT: 更长上下文，结构区域加权，注释感知更强。
- 对 PlantCAD2: 若 PlantCAD2 训练数据更广，本项目不保证全任务超过；但在本地作物结构注释任务和严格 holdout 上应更贴合。
- 对 CNN/DeepSEA-like: 跨物种、小样本、gene family holdout 应明显更强。

## 10. 进展记录

- 2026-06-07 23:26:04 CST: 完成第一版 CropGenome-FM 模型结构定义，确定主线为单碱基、长上下文、RC 等变双向 Mamba/Hyena 架构，参数档位为 Base/Large/XL，第一正式目标为 Large。
- 2026-06-07 23:46:31 CST: 增补 2026 前沿架构取舍，明确主模型为 RC-equivariant Mamba2/Hyena + periodic attention 的混合结构；补充输入张量、输出张量、训练目标、显存策略和分阶段资源估算。
- 2026-06-08 11:42:31 CST: 按用户要求改为只使用 262 个结构注释完整基因组，加入 region_ids、region_weights、区域加权 loss、片段过滤和基线优势预期。
- 2026-06-08 15:00:51 CST: 明确最终训练数据管线不采用进一步压缩到核心 assembly 的方案。
- 2026-06-08 18:45:35 CST: 区域采样比例参考 `douke_genome`，加入 TE/repeat 注释模式和当前默认无 TE fallback 模式。
- 2026-06-08 19:15:22 CST: 输入侧加入 4.5 硬质量过滤、候选池保留比例和去冗余控制；模型训练仍按 batch 目标区域比例和 loss 权重动态采样。
- 2026-06-08 22:19:54 CST: 输入管线更新为本服务器固化 stage 输入，训练服务器动态生成 mask/label/RC；新增 `training_server_transfer/` 作为跨服务器传输目录。
