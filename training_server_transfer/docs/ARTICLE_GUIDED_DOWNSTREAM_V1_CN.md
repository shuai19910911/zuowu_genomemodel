# 论文引导的 CropGenome-FM 下游评估 v1（中文说明）

## 1. 当前状态

- 所有下游 GPU/CPU worker、daemon 和旧完成监控器均已停止。
- `FORMAL_EXECUTION_AUTHORIZATION.json` 已置为 `paused_for_article_guided_downstream_redesign`；当前不能被旧调度器自动拉起。
- Stage C1 与 EDTA 数据处理不属于下游执行，继续保留。
- 本文档只冻结设计与代码入口，不宣称新下游结果已经产生。

## 2. 两篇文章给出的直接证据

### 2.1 Nature Communications 2025

论文：**Benchmarking DNA foundation models for genomic and genetic tasks**。Nature Communications 16, 10780 (2025)。DOI：[10.1038/s41467-025-65823-8](https://doi.org/10.1038/s41467-025-65823-8)。

原文与官方资源：

- 论文：https://www.nature.com/articles/s41467-025-65823-8
- 官方代码：https://github.com/ChongWuLab/dna_foundation_benchmark
- 官方处理后数据：https://huggingface.co/datasets/hfeng3/dna_foundation_benchmark_dataset

可直接采用的证据：

1. 论文比较 9 个 DNA foundation models，在 57 个短序列分类数据集、GTEx 表达、variant effect（变异效应）、TAD 和运行时任务上统一重跑。
2. 分类任务统一比较 frozen embedding（冻结向量）而不是各模型各自微调；主要报告 AUROC/accuracy，并用训练集内 5-fold CV 选头部参数。
3. pooling（池化）不是无关细节。mean pooling 在 56.6% 的 model-task 组合中最好，max pooling 为 38.2%，special token 仅 3.4%。因此本项目锁定 masked mean 为主协议，不默认把每个任务重复跑三次。
4. 变异效应采用 `embedding(alt)-embedding(ref)`，并按染色体外层 holdout、训练染色体内层 CV，避免 LD 和邻近序列泄漏。本项目保留 D02/D03 和 C28，并继续使用 chromosome/population holdout。
5. 短序列结果不能回答长上下文是否有用。论文另做 6 kb 与 196 kb 对照，但不同模型实际收到的长度不同，且 long track 只覆盖部分模型。我们的 32K/64K track 必须独立于 common-context 主榜。
6. 官方实现把全部 embedding 放入 Python list/DataFrame 后再拼接，适合复现论文，但不适合 192 GiB 主机上的多 2080Ti 并发。本项目只借鉴评估思想，不照搬其内存实现。

不直接采用的部分：

- 论文大量任务来自人类和鼠。本项目主证据严格限定 plant/crop DNA-only，不把人类 GTEx、ClinVar、TAD 数据混入作物主榜。
- Arabidopsis 4mC 数据在论文中被明确提醒：部分真核 4mC 标签依赖计算预测，而非统一的高严格度化学测定；暂不把它加入正式主榜。
- 论文短任务有些采用随机 70:30 重分割；本项目优先保留 chromosome/species/genus/assembly fixed test，不回退到容易泄漏的随机 split。

### 2.2 GENEB / arXiv 2026

论文：**GENEB: A Unified and Comprehensive Benchmark for DNA Sequence Classification**，arXiv:2606.04525（v1，2026-06-03）。原文：https://arxiv.org/abs/2606.04525 。

可直接采用的证据：

1. GENEB 汇总 100 个任务、13 个功能类别，统一使用 frozen embedding + logistic regression，并以 MCC（Matthews correlation coefficient，马修斯相关系数）比较模型。
2. 主实验把超大任务限制到最多 100,000 条序列，并说明多数任务在 50,000 条附近趋于稳定。本项目只限制 training split 到 100,000；validation/test 永不抽样，保留 fixed-test 完整性。
3. 1-shot、10-shot、full-data 均使用固定 seeds `[13,17,42,123,997]`；本项目按相同 seeds 增加 few-shot 结果。
4. 正则化敏感性覆盖 `C=[0.01,0.1,1,10,100]`；本项目线性分类头使用同一网格。
5. GENEB 同时报告 category macro average（类别宏平均）与 task micro average（任务微平均），并定义 specialization score：类别外平均 rank 减去类别内平均 rank。只报一个总平均会掩盖模型专长，因此本项目新增类别汇总与 specialization。
6. GENEB 把 mean MCC < 0.35 定义为 hard task，把跨模型 MCC 标准差 > 0.12 定义为 high-variance task。本项目按相同阈值生成诊断清单。
7. GENEB 的迁移分析对本项目很关键：在人类基因组上预训练的模型迁移到植物任务时可出现负迁移；多物种预训练通常更好，但并非所有任务都提升。因此植物/作物公开模型和通用 DNA 模型必须并列真实重跑，不能只比人类强模型。
8. GENEB 明确排除了 >10 kb long-range interaction 任务；所以它不能替代 CropGenome-FM 的长上下文证据。

限制：

- arXiv v1 页面没有给出可核验的代码或完整数据仓库链接；本项目只采用论文中可复述、可机器实现的 protocol，不声称复现了 GENEB 原始代码。
- GENEB 的 100 任务以分类为主，而本项目还包括表达回归、候选排序和零样本变异效应；这些任务保留各自原生主指标，MCC 只用于分类汇总。

## 3. 冻结后的任务矩阵

默认 article-guided profile 共 25 项：A+B 的 17 项核心任务全部保留，再增加 8 项植物专属补充任务。相比旧 registry 直接展开 53 项，任务数减少 52.83%，避免重复来源和大量不进入主证据的 parity/fine-tuning 任务占用 2080Ti。

### 3.1 核心 17 项

| 类别 | Task IDs | 内容 |
|---|---|---|
| splice/promoter/poly(A) | A01-A03 | 剪接、启动子、转录终止 |
| lncRNA/enhancer | A04-A05 | 多物种 lncRNA、增强子 |
| expression | A06-A10 | Arabidopsis、水稻、玉米、小麦、大豆表达回归 |
| TE | B11-B12 | TE order/family |
| gene structure | B13-B16 | gene architecture、候选排序、enhancer ranking、exon order |
| low homology | B17 | 低同源敏感性分析 |

B11/B12 仍属于完整 25 项；在 EDTA 数据完成前，正式 Gate 应拒绝把部分矩阵称为完整结果。

### 3.2 论文引导补充 8 项

| Task ID | 类别 | 加入原因 |
|---|---|---|
| C05 | 7-species chromatin accessibility | GENEB 的 plant accessibility 类别；跨物种泛化 |
| C08-C10 | H3K27ac/H3K27me3/H3K4me3 | 植物表观遗传类别；比新增标签存疑的 4mC 更可靠 |
| C28 | structural variant effect | Nature 的 alt-ref effect vector 思路 |
| D01 | Arabidopsis 7-region chromosome holdout | 固定 chromosome test，拒绝随机 split |
| D02 | rare/common variant discrimination | 变异频率与作物群体证据 |
| D03 | AraGWAS hit enrichment | 作物/植物 GWAS 零样本证据 |

明确不加入主矩阵：人/鼠任务、no-region/random-init 内部消融、单独 fine-tuning parity 任务，以及缺少可靠测量来源的 Arabidopsis 4mC。

## 4. 新评估协议

1. 用户每次只指定一个 checkpoint；禁止在同一正式矩阵中同时试多个 checkpoint 后看 test 选最优。
2. classification 主汇总用 MCC，同时保留每项原生指标（AUROC、macro-F1 等）。
3. fixed seeds：`13,17,42,123,997`。
4. full-data：训练集内选择 `C=[0.01,0.1,1,10,100]`；test 只做报告。
5. few-shot：仅从 training split 每类抽 1/10 条，C 固定为 1；validation/test 不抽样、不用于选样。
6. training split 超过 100,000 时做确定性分层截取；validation/test 永不截取。
7. 默认 pooling 为 masked mean。max/special-token 是 post-matrix 可选敏感性，不默认把 GPU 工作量放大三倍。
8. 默认只运行已有 common-context rows。32K/64K long track 当前关闭；必须先在 2080Ti 对每个模型做真实显存 profile，再单列扩展榜，不能和 common-context 混成一个主榜。
9. 分类结果必须同时输出 category macro、task micro、specialization、hard tasks 和 high-variance tasks。
10. 矩阵未齐时 `final_leaderboard_allowed=false`。

## 5. RAM/GPU 优化

已经落地的执行优化：

- embedding 从“全任务 float32 list → concatenate → float16 副本”改成逐 batch 写 float16 memmap，再原子生成缓存。
- 合成基准 `131072×512` 的峰值 RSS 从 690,380 KiB 降到 183,752 KiB，降低 73.38%（3.76 倍）。
- CropGenomeFM 在 GPU 上先转 float16 再回传 CPU，减少 PCIe 和主机副本。
- 短 context 按 token budget 放大 batch，OOM 时自动减半；真实 GPU2 烟测中 Caduceus batch 2→16 的吞吐从 76.31 提高到 262.82 samples/s，约 3.44 倍；最大 embedding 差为 `3.05e-05`，cosine 最低 `0.99999988`。
- GPU daemon 默认最多 3 个 worker、每卡 1 项、禁止 foreign compute；本机 CPU probe daemon 不与 GPU embedding 同时自动启动。
- host RAM 按完整子进程树 RSS 记账，任务超过预算时主动终止并留回执。
- gpu05 的 `nvidia-smi --query-compute-apps` 会被残留上下文拖入不可中断等待；memory-packed 模式改为 basic memory/utilization 查询、5 秒共享缓存和 8 MiB unknown-memory 阈值。实测基础查询 8.16 秒，但多个 worker 只触发一次，不再串行放大。
- 文章 profile 从 53 项收缩到 25 项，并对超大训练集封顶 100,000，进一步降低物理内存和重复 GPU 开销。

GPU 利用率百分比未在原环境中直接采到：该环境无 pynvml，旧 nvidia-smi 采样过慢。这里报告真实吞吐，不伪造 utilization 数值。

## 6. 指定 checkpoint 的入口

只准备、审计，不占 GPU：

```bash
/home/user/zhangzhishuai/.local/share/mamba/envs/zuowu_genomemodel/bin/python \
  scripts/launch_cropgenome_downstream_checkpoint.py \
  --checkpoint /absolute/path/to/step_00040000.pt
```

真实启动（仅在你明确指定 checkpoint 后运行）：

```bash
/home/user/zhangzhishuai/.local/share/mamba/envs/zuowu_genomemodel/bin/python \
  scripts/launch_cropgenome_downstream_checkpoint.py \
  --checkpoint /absolute/path/to/step_00040000.pt \
  --execute \
  --confirmation ACTIVATE_ARTICLE_GUIDED_CHECKPOINT
```

该命令会：

1. 锁定 checkpoint 路径、step、size、mtime 和 SHA256；
2. 生成单 checkpoint、25 任务 plan；
3. 检查当前下游是否已停止；
4. 归档旧 plan/authorization；
5. 完整正式 Gate 通过后才授权；
6. 以 gpu05、3 workers、每卡 1 项启动；
7. 任一 Gate 或启动步骤失败则恢复旧 plan/authorization。

真实 step40000 的 prepare-only 已运行成功，候选 plan：

`training_server_transfer/runs/cropgenome_downstream_final_v1/controller/checkpoint_plans/article_step_00040000.json`

当前没有执行 `--execute`，符合“停止全部下游任务”的要求。

当前真实数据 Gate：25 项中 23 项无需 EDTA；B11/B12 仍等待 EDTA 规范化结果，B17 是低同源敏感性而非独立数据集。因此在 B11/B12 ready 前，`--execute` 会 fail-closed，不会把 23/25 或 22/24 称为完整矩阵。

## 7. 主要代码

- `training_server_transfer/configs/downstream_article_guided_v1.json`：25 项任务与文章协议。
- `training_server_transfer/downstream_v4/article_protocol.py`：单 checkpoint plan builder。
- `training_server_transfer/downstream_v4/checkpoint_launcher.py`：checkpoint identity 与安全命令生成。
- `scripts/launch_cropgenome_downstream_checkpoint.py`：prepare/execute 入口。
- `training_server_transfer/downstream_v4/probe.py`：MCC、GENEB seeds、1/10-shot、训练 cap。
- `training_server_transfer/downstream_v4/article_analysis.py`：macro/micro、specialization、hard/high-variance。
- `scripts/summarize_cropgenome_article_results.py`：矩阵完成后的机器汇总。
- `training_server_transfer/downstream_v4/streaming_embeddings.py`：低 RAM embedding 缓存。

## 8. 当前不能说明什么

- 还没有 article-guided 25 项的新模型结果，不能说新 checkpoint 优于任何公开模型。
- 旧结果不能自动改名为新协议结果。
- 32K/64K 长程收益没有在 2080Ti 和固定 crop test 上完成验证，当前不进入主榜。
- GENEB 是 arXiv v1，尚不能替代同行评议证据；它提供的是统一 probe 设计启发。
- throughput 提升不等于所有模型/所有 context 的利用率都同比提高，后续必须在指定 checkpoint 的 smoke 中逐模型记录。
