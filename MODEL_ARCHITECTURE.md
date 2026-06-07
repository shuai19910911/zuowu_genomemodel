# CropGenome-FM 模型结构解析

更新时间: 2026-06-07 23:46:31 CST

## 1. 设计结论

主模型采用长上下文单碱基模型，而不是 6-mer 短窗口 BERT 复刻。原因是作物基因组任务同时需要:

- SNP/indel 级别的单碱基分辨率。
- promoter、enhancer、gene body、terminator、结构变异等长距离上下文。
- DNA 双链反向互补一致性。
- 跨属、跨物种、跨品种迁移。

因此主架构定义为:

CropGenome-FM = single-nucleotide tokenizer + RC-equivariant bidirectional sequence backbone + long-context curriculum pretraining + PEFT/downstream heads。

## 2. 输入表示

### 2.1 Token vocabulary

基础词表:

- `A`, `C`, `G`, `T`
- `N`
- `[PAD]`, `[MASK]`, `[BOS]`, `[EOS]`
- 可选: lowercase/softmask 标记作为 side-channel，不并入第一版主词表。

单碱基 token 的优势是变异效应评分天然对齐 ref/alt，避免 BPE 或 k-mer 在单点突变附近造成 token 边界变化。

### 2.2 序列增强

- reverse complement augmentation: 每个窗口以 50% 概率输入反向互补。
- random shift/crop: 防止模型记住固定窗口边界。
- N-aware masking: N 区域不作为预测目标，但可作为上下文保留。
- assembly/genus balanced sampling: 防止高频属支配训练。

## 3. Backbone

### 3.1 推荐主干

第一选择: Caduceus/PlantCAD 风格的 RC-equivariant bidirectional Mamba2。

第二选择: Evo 2/StripedHyena 2 风格的 Hyena + attention 多混合 block。

第三选择: HybriDNA 风格的 Transformer-Mamba2 hybrid decoder，同时保留理解任务和生成式 likelihood 能力。

工程上优先使用已有可靠实现作为训练骨架，而不是手写全新 state-space kernel。选择标准:

- 支持 bf16。
- 支持 FSDP/DeepSpeed。
- 支持 >= 128K context 的稳定训练。
- 支持自定义 DNA tokenizer 和 MLM/span denoising head。
- 可导出 HuggingFace 风格 checkpoint 或至少有清晰 inference API。

### 3.1.1 前沿架构对比

| 方向 | 代表 | 优点 | 局限 | 本项目取舍 |
|---|---|---|---|---|
| 6-mer Transformer MLM | AgroNT, Nucleotide Transformer | 稳定、容易微调、植物任务已有强基线 | 上下文短，单点变异 token 边界不自然 | 作为基线，不作为主模型 |
| BPE DNA Transformer | DNABERT-2, GROVER | token 更少，训练效率高 | 变异效应解释受 tokenization 影响 | 作为效率对照，可后续做 BPE 分支 |
| 单碱基 Hyena | HyenaDNA, Evo 2 | 长上下文、单碱基、适合 likelihood | 工程实现复杂，训练调参要求高 | 主路线重要组成 |
| RC 等变 Mamba | Caduceus, PlantCAD | DNA 双链归纳偏置强，植物模型已有先例 | 原始 PlantCAD 输入较短，PlantCAD2 仍需复现细节 | 主模型默认归纳偏置 |
| Transformer-Mamba2 hybrid | HybriDNA | 注意力捕获精确局部/中程交互，Mamba2 处理长程 | 显存和实现复杂度高 | 在每 4-6 层插入 attention 的推荐结构 |
| 监督多任务长模型 | AlphaGenome/Enformer/Borzoi | 功能基因组预测强 | 依赖大规模标签，人/鼠为主 | 作为下游 head 和评估思想，不直接照搬 |

结论: CropGenome-FM-Large 采用 `RC-equivariant Mamba2/Hyena backbone + periodic local/global attention + dual MLM/causal objective`。这比纯短窗口 BERT 更接近 2026 年前沿基因组基础模型方向，同时保留植物 DNA 双链归纳偏置。

### 3.2 Block 结构

每层包含:

1. RMSNorm。
2. Bidirectional Mamba 或 Hyena operator。
3. Gated MLP。
4. Residual connection。
5. Dropout/DropPath 只在必要时开启，正式预训练优先稳定。

RC 等变处理:

- 输入序列同时存在 forward 与 reverse-complement 对称视角。
- hidden channels 按 forward/reverse 分组。
- LM head 对 RC token 做一致性约束。
- 下游 embedding 默认输出 forward 与 reverse-complement 平均后的表示。

## 4. 参数规模

| 配置 | 层数 | hidden | heads/通道组 | 估计参数 | 初始上下文 | 长上下文 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 24 | 768 | 12 或等价通道组 | 100M-150M | 8192 | 65536/131072 |
| Large | 32 | 1024 | 16 或等价通道组 | 300M-450M | 8192 | 131072/262144 |
| XL | 40 | 1536 | 24 或等价通道组 | 800M-1.2B | 8192 | 262144/524288/1M |

本项目第一主目标是 Large；Base 只作为资源不足时的正式较小模型，不能当作测试模型描述。

## 5. 预训练目标

### 5.1 主目标: masked nucleotide/span denoising

- mask rate: 15% token/span。
- span length: 几何分布，覆盖 1 bp 到数百 bp。
- 对 N 和低复杂度区域降低采样为预测目标的概率。
- loss: cross entropy over A/C/G/T/N，主报告只统计 A/C/G/T 有效位点 loss。

### 5.2 辅助目标

可选加入:

- reverse-complement consistency loss。
- next-token 或 prefix LM auxiliary loss，用于长上下文生成式打分。
- masked segment reconstruction，用于 indel/SV 敏感表示。

### 5.3 输入到输出的精确定义

输入:

```text
raw_sequence:       FASTA 子串，例如 chr1:100000-108191
canonical_sequence: A/C/G/T/N，大写；其他 IUPAC 碱基归为 N 或按配置随机投影
input_ids:          [B, L], L = 8192/65536/131072/262144
labels:             [B, L], 非 mask 位置为 -100
metadata:           assembly_id/species_id/genus_id/contig/start/end/strand
```

训练输出:

```text
logits_mlm:         [B, L, vocab_size]
hidden_states:      [B, L, hidden]
sequence_embedding: mean/attention/center pooled embedding
loss_mlm:           masked nucleotide cross entropy
loss_rc:            reverse-complement consistency loss
loss_causal:        optional next-token loss for likelihood scoring
```

推理输出:

- 序列 embedding。
- 每碱基 likelihood。
- ref/alt delta likelihood。
- 下游任务 logits 或 regression value。

## 6. 训练课程

| 阶段 | context | 数据 | 目标 |
|---|---:|---|---|
| Stage B | 8192 | 全部 1906 genome 随机窗口 | 学习局部 motif、codon、splice signal、重复序列 |
| Stage C1 | 65536 | scaffold/chromosome 连续窗口 | 学习 gene-proximal 与调控上下文 |
| Stage C2 | 131072 | 长窗口混合采样 | 学习远端调控和结构上下文 |
| Stage D | 262144+ | 高质量长 scaffold/chromosome | 长上下文 midtraining |

### 6.1 GPU 内存策略

- 8K: bf16 + activation checkpointing；Large 可在 2-4 张 80GB GPU 上训练。
- 64K/128K: 必须使用 FSDP/ZeRO-3、sequence chunking、activation checkpointing；建议 8 张 80GB GPU。
- 256K+: 建议 8-16 张 80GB GPU；若资源不足，先完成 128K 正式模型，不强行牺牲 batch 和稳定性。
- optimizer: AdamW 或 Lion；正式首选 AdamW，`betas=(0.9,0.95)`，weight decay 0.1，cosine decay + warmup。
- gradient clipping: 1.0。
- precision: bf16；不建议 fp16。

### 6.2 资源估算

| 模型 | context | 推荐 GPU | 单步全局 token | 训练 token | 预计训练时间 |
|---|---:|---:|---:|---:|---:|
| Large Stage B | 8K | 4-8 x 80GB | 0.5M-2M | 50B-150B | 5-25 天 |
| Large Stage C | 64K/128K | 8 x 80GB | 0.25M-1M | 20B-80B | 7-21 天 |
| Large Stage D | 256K | 8-16 x 80GB | 0.125M-0.5M | 5B-30B | 4-21 天 |
| Base fallback | 8K/64K | 2-4 x 80GB | 0.5M-2M | 30B-80B | 2-6 周 |

这些估算必须在首个 1000 step 后用真实 tokens/s 校正。文档中的时间是规划上限，不替代实测吞吐。

训练过程中保存:

- optimizer state checkpoint。
- inference checkpoint。
- tokenizer/config。
- training manifest hash。
- validation loss by genus/species/assembly。

## 7. 下游 head

### 7.1 分类任务

- 输入: mean pooling、center-token pooling、attention pooling 三种都保留。
- head: 2 层 MLP。
- 任务: promoter、enhancer/open chromatin、splice donor/acceptor、TIS/TTS、lncRNA/mRNA。

### 7.2 回归任务

- 输入: promoter-proximal embedding 或 gene-region pooled embedding。
- head: MLP/regression。
- 任务: promoter/terminator strength、gene expression proxy、chromatin accessibility intensity。

### 7.3 变异效应

三种评分同时输出:

- delta log-likelihood: `LL(alt) - LL(ref)`。
- embedding delta: `embedding(alt) - embedding(ref)`。
- fine-tuned variant head: 对已知功能变异做 LoRA/IA3 微调。

## 8. 评估基线

必须比较:

- CNN/DeepSEA-like 从头训练模型。
- DNABERT-2。
- AgroNT。
- PlantCAD/PlantCAD2。
- HyenaDNA/Caduceus 可用 checkpoint。

比较方式:

- zero-shot embedding + random forest/logistic regression。
- LoRA/IA3 参数高效微调。
- full fine-tune。
- species/genus holdout。

## 9. 风险和控制

- 数据泄漏: assembly/species/genus 分层切分，窗口不跨 split。
- 属不平衡: 采样权重和分属 validation loss。
- 注释质量不均: 下游任务记录 annotation source 和 assembly level。
- 模型只学 GC/长度: 负例按长度和 GC 匹配。
- 长上下文显存不足: 课程式扩展 context，gradient checkpointing，FSDP/ZeRO-3。
- 预训练 loss 与生物任务脱节: 每阶段都做轻量但正式的下游 probe。

## 10. 进展记录

- 2026-06-07 23:26:04 CST: 完成第一版 CropGenome-FM 模型结构定义，确定主线为单碱基、长上下文、RC 等变双向 Mamba/Hyena 架构，参数档位为 Base/Large/XL，第一正式目标为 Large。
- 2026-06-07 23:46:31 CST: 增补 2026 前沿架构取舍，明确主模型为 RC-equivariant Mamba2/Hyena + periodic attention 的混合结构；补充输入张量、输出张量、训练目标、显存策略和分阶段资源估算。
