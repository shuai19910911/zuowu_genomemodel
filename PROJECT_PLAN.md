# 作物基因组预训练大模型详细训练方案

更新时间: 2026-06-07 23:26:04 CST

## 1. 目标

训练一个面向作物基因组的正式基础模型，不做玩具版、抽样测试版或非正式小模型。模型应直接服务于作物基因组功能解析和育种相关任务:

- 学习多作物、多属、多品种 genome 的通用序列表示。
- 在未见作物或少样本作物上迁移到功能区间识别、基因结构相关任务和变异效应预测。
- 支持零样本序列打分、参数高效微调、全模型微调和批量 embedding 生成。
- 为后续作物基因组调控元件、候选变异、基因编辑靶点优先级排序提供基础模型。

## 2. 文献依据

当前证据支持以下设计取舍:

- AgroNT 是最直接的植物参照。它在 48 个植物物种 genome 上训练 transformer DNA LM，使用约 6000 bp 输入、6-mer token、15% MLM，并在植物调控注释、promoter/terminator 强度、组织特异表达和功能变异优先级等任务上取得强结果。来源: https://www.nature.com/articles/s42003-024-06465-2
- PlantCAD/PlantCaduceus 是当前更贴近本项目架构需求的植物模型路线，基于 Caduceus/Mamba，强调双向、反向互补等变和单碱基分辨率；PlantCAD 预训练 16 个被子植物 genome，PlantCAD2 提供 8192 bp、88M/311M/694M 模型。来源: https://github.com/plantcad/plantcad
- Evo 2 说明基因组基础模型的正式长上下文训练趋势: 先短上下文预训练，再长上下文 midtraining；模型覆盖 9T DNA bp，目标上下文达 1M token。来源: https://www.nature.com/articles/s41586-026-10176-5
- HyenaDNA 证明单碱基、长上下文 genome modeling 可行，并将上下文扩到 1M token。来源: https://arxiv.org/abs/2306.15794
- DNABERT-2 和 GROVER 说明 DNA tokenization 不能只机械依赖固定 k-mer；BPE 能提高计算效率，但长上下文单碱基模型更适合保留变异级分辨率。来源: https://arxiv.org/abs/2306.15006 和 https://www.nature.com/articles/s42256-024-00872-0
- 2025 年 DNA foundation model benchmark 显示，通用 foundation model 在不同任务上的优势并不均匀；必须同时做零样本 embedding、参数高效微调和任务特化微调评测，不能只看预训练 loss。来源: https://www.nature.com/articles/s41467-025-65823-8

## 3. 数据策略

### 3.1 数据来源

本地数据目录: `/home/user/zhangzhishuai/data/plantDB/genome`

- 1906 个含 genome FASTA 的 assembly 用于自监督预训练。
- 262 个含 genome + GFF3/GTF 的 assembly 用于下游监督任务构建、预训练评估和 gene-region aware 数据切分。
- 当前不把 Phytozome 数据纳入正式训练，除非后续完成可复现下载、许可和版本记录。

### 3.2 预训练数据清洗

必须先在当前项目目录生成可复现 manifest，不直接改动原始数据目录:

1. 扫描 `completed-genome-index.tsv` 与 `incomplete-genome-index.tsv`，生成训练 manifest。
2. 过滤短 contig、异常字符比例高的序列、过高 N 比例窗口。
3. 保留 soft-mask/lowercase 信息作为可选辅助通道；主输入仍为 A/C/G/T/N 单碱基。
4. 按 assembly、species、genus 分层划分 train/val/test，严禁同一 assembly 的相邻窗口跨 split 泄漏。
5. 对多拷贝、重复、近重复 genome 做 MinHash/seqkit 级别去冗余记录，但不盲目删除作物品种内真实变异；去冗余策略作为数据权重而非简单丢弃。

### 3.3 采样权重

作物数据按属极不均衡，Beta、Hordeum、Oryza、Zea 等数量高。预训练采样采用混合权重:

- 50% assembly-balanced: 防止大属和大基因组完全支配。
- 30% genus-balanced: 提升小属、孤儿作物表示。
- 20% length-proportional: 保留真实 genome bp 分布。

训练时记录每个 batch 的 assembly/species/genus 统计，后续评估是否过度偏向高频属。

## 4. 主模型训练路线

### 4.1 主模型名称

临时名称: CropGenome-FM。

### 4.2 主架构

正式主线采用 Caduceus/PlantCAD 启发的双向 RC-equivariant Mamba/Hyena 混合模型:

- token: 单碱基 A/C/G/T/N，加 special tokens。
- input resolution: 1 bp/token。
- core: bidirectional Mamba/Hyena blocks。
- reverse-complement equivariance: forward strand 与 reverse-complement strand 共享或成对约束参数，在输出端做 RC-consistent LM head。
- pretraining objective: masked nucleotide modeling + span denoising；长上下文阶段加入 autoregressive/next-token auxiliary loss 作为可选辅任务。
- precision: bf16。
- distributed training: PyTorch FSDP 或 DeepSpeed ZeRO-3，优先选择集群可稳定运行的一套。

### 4.3 参数规模

第一版正式模型不做玩具规模，采用三档可发表/可复用规模:

| 模型 | 层数 | hidden | 估计参数 | 上下文 | 用途 |
|---|---:|---:|---:|---:|---|
| CropGenome-FM-Base | 24 | 768 | 100M-150M | 64K -> 128K | 主力可训练、可微调模型 |
| CropGenome-FM-Large | 32 | 1024 | 300M-450M | 128K -> 256K | 正式主模型 |
| CropGenome-FM-XL | 40 | 1536 | 800M-1.2B | 256K -> 512K/1M | 资源允许后续扩展 |

推荐先直接启动 Large 的正式数据管线和正式训练配置；Base 只作为资源不足时的正式较小模型，不作为测试版。

### 4.4 训练阶段

Stage A: 数据 manifest 与 tokenizer/编码确认。

- 产物: `data_manifests/`、窗口统计、split 表。
- CPU: q07/q08，30 核，150G 内存。
- 判定: 100% 可追溯到原始 genome 路径和 assembly accession。

Stage B: 8K 基础预训练。

- 输入长度: 8192 bp。
- batch: 按 GPU 显存确定，使用 gradient accumulation 保持全局 batch >= 1M bp。
- 目标: MLM/span denoising loss 稳定下降，val split 不泄漏。

Stage C: 64K/128K 继续预训练。

- 从 Stage B checkpoint 继续。
- 增加长窗口和跨 gene/intergenic 长程上下文。
- 目标: 长上下文 validation loss 下降，embedding 下游线性探针优于随机/CNN baseline。

Stage D: 256K 长上下文 midtraining。

- 借鉴 Evo 2 的短到长课程训练。
- 以 chromosome/scaffold 连续窗口为主。
- 目标: 变异打分、调控区间和基因结构任务在长上下文输入中收益可见。

Stage E: 下游微调与基准。

- 参数高效微调: LoRA/IA3。
- 全模型微调: 只在关键任务上跑。
- 零样本: ref/alt log-likelihood ratio、mean embedding、variant effect vector。

## 5. 下游任务设计

优先使用本地 262 个完整注释 genome 构建:

1. Splice donor/acceptor: 从 GFF3/GTF 提取 exon-intron junction，构造同染色体 hard negatives。
2. TIS/TTS: CDS start/stop 周围窗口，按 gene family 或 chromosome split 防泄漏。
3. Promoter/terminator: TSS 上游/下游窗口；若没有表达强度标签，先做区域分类，后续接公开 STARR-seq/表达数据。
4. lncRNA/mRNA: 用注释 transcript，按长度和 GC 匹配负例，避免模型只学长度。
5. Chromatin/open region: 接入公开植物 ATAC-seq peak 数据，优先 rice/maize/Arabidopsis/soybean。
6. Tissue expression: promoter-proximal sequence 到表达量或表达/不表达分类。
7. Variant effect: ref/alt 序列 log-likelihood ratio、embedding difference、LoRA 分类器；优先 rice、maize、soybean 已知农艺变异。

关键评估切分:

- Random split 只作辅助。
- Assembly holdout。
- Species holdout。
- Genus holdout。
- Crop-family holdout。

## 6. 预计结果

正式训练成功的最低标准:

- 预训练 validation loss 在同 split 持续下降，无明显数据泄漏。
- 在 splice/TIS/TTS/promoter/open chromatin 至少 3 类任务上，微调结果超过从头训练 CNN baseline。
- 在 species/genus holdout 中，优于 DNABERT-2/AgroNT/PlantCAD 可取得 checkpoint 的零样本 embedding 或 LoRA 微调基线中的至少一部分。
- 对 ref/alt 变异能输出稳定、方向一致的 log-likelihood 或 embedding effect score。

高价值结果:

- 在作物特异任务上优于 AgroNT 或 PlantCAD 同规模/同输入长度模型。
- 在小样本作物迁移和孤儿作物任务上有明显收益。
- 长上下文阶段相对 8K 模型在表达、调控或变异任务上带来可量化提升。

## 7. 当前环境和执行原则

- 登录节点不直接跑重 CPU/GPU 任务。
- CPU 作业生成 `sbatch -p q07 -c 30 ...` 或 `sbatch -p q08 -c 30 ...` 脚本，由用户或后续明确步骤提交。
- GPU 命令只生成，不自动执行，例如 `CUDA_VISIBLE_DEVICES=1,2 python train.py ...`。
- 所有新脚本、manifest、日志、配置、结果都放在当前目录下。
- GitHub 只同步 `README.md`、`PROJECT_PLAN.md`、`MODEL_ARCHITECTURE.md` 这类轻量项目文档。

## 8. 近期里程碑

- M1: 生成训练 manifest、split 和窗口统计。
- M2: 建立正式数据读取器和 batch 采样权重配置。
- M3: 安装/固定训练依赖，选择 FSDP 或 DeepSpeed。
- M4: 生成 Large 模型正式预训练配置和用户可执行 GPU 命令。
- M5: 完成 8K 预训练阶段。
- M6: 完成 64K/128K 继续预训练阶段。
- M7: 完成核心下游任务数据集与评测。

## 9. 进展记录

- 2026-06-07 23:26:04 CST: 读取 `/home/user/zhangzhishuai/data/plantDB/genome/README.md`，确认训练数据口径；完成 AgroNT、PlantCAD/Caduceus、Evo 2、HyenaDNA、DNABERT-2、GROVER 和 DNA foundation benchmark 调研；确定主路线为长上下文、单碱基、RC 等变双向 Mamba/Hyena 模型。

