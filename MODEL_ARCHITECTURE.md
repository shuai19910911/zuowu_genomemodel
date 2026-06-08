# CropGenome-FM 模型结构解析

更新时间: 2026-06-08 19:15:22 CST

## 1. 设计结论

CropGenome-FM 是一个面向作物结构注释基因组的区域加权长上下文基础模型。第一版正式模型只使用 262 个有 FASTA + GFF3/GTF 的 assembly，不使用缺少结构注释的 genome。训练数据采用在线采样和在线 tokenization，不预生成全量 token shards。

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
- 2026-06-08 15:00:51 CST: 明确最终训练数据管线为原始压缩数据 + 小索引 + 在线采样/tokenization + 100-200GB 磁盘缓存，不采用进一步压缩到核心 assembly 的方案。
- 2026-06-08 18:45:35 CST: 区域采样比例参考 `douke_genome`，加入 TE/repeat 注释模式和当前默认无 TE fallback 模式。
- 2026-06-08 19:15:22 CST: 输入侧加入 4.5 硬质量过滤、候选池保留比例和去冗余控制；模型训练仍按 batch 目标区域比例和 loss 权重动态采样。
