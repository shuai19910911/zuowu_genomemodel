# CropGenome-FM 模型结构解析

更新时间: 2026-06-07 23:26:04 CST

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

第一选择: Caduceus/PlantCAD 风格的 RC-equivariant bidirectional Mamba。

第二选择: HyenaDNA/Evo 2 风格的长卷积/Hyena 混合 block。

工程上优先使用已有可靠实现作为训练骨架，而不是手写全新 state-space kernel。选择标准:

- 支持 bf16。
- 支持 FSDP/DeepSpeed。
- 支持 >= 128K context 的稳定训练。
- 支持自定义 DNA tokenizer 和 MLM/span denoising head。
- 可导出 HuggingFace 风格 checkpoint 或至少有清晰 inference API。

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

## 6. 训练课程

| 阶段 | context | 数据 | 目标 |
|---|---:|---|---|
| Stage B | 8192 | 全部 1906 genome 随机窗口 | 学习局部 motif、codon、splice signal、重复序列 |
| Stage C1 | 65536 | scaffold/chromosome 连续窗口 | 学习 gene-proximal 与调控上下文 |
| Stage C2 | 131072 | 长窗口混合采样 | 学习远端调控和结构上下文 |
| Stage D | 262144+ | 高质量长 scaffold/chromosome | 长上下文 midtraining |

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

