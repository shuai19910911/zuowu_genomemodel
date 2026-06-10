# training_server_transfer 目录结构和搬运说明

更新时间: 2026-06-10 18:04:57 CST

## 当前结论

`training_server_transfer/` 已整理为训练服务器唯一需要搬运的简约目录。训练服务器不需要访问本服务器原始 genome 目录，也不需要搬运本服务器的大型中间索引。

搬运目录:

```text
training_server_transfer/
```

搬运后先校验:

```bash
cd training_server_transfer
sha256sum -c SHA256SUMS
```

当前已按“主 context 长度候选全部保留 + 其他长度受控回放 + 更严格窗口质量过滤”的新策略完成输入生成。最终目录约 `67G`，已包含简洁训练脚本目录 `scripts/`、正式训练配置和 `zuowu_genomemodel` 环境导出文件；全目录 SHA256 已刷新并通过校验。

## 简约目录结构

```text
training_server_transfer/
  README.md
  MANIFEST.tsv
  SHA256SUMS
  configs/
    model_large.json
    stage_B_mix.yaml
    stage_C1_mix.yaml
    stage_C2_mix.yaml
    stage_D_mix.yaml
    train_stage_B.json
    train_stage_C1.json
    train_stage_C2.json
    train_stage_D.json
    zuowu_genomemodel_env.yml
  metadata/
    assemblies.canonical.tsv
    assembly_splits.canonical.tsv
    assembly_splits.canonical.summary.tsv
    annotation_qc.summary.tsv
    fasta_qc.summary.tsv
    region_candidates.summary.tsv
    seqid_alias.summary.tsv
    token_vocab.tsv
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
  scripts/
    README.md
    check_package.py
    requirements.txt
    run_stage.sh
    train.py
```

## 目录说明

### `configs/`

保存模型结构配置、训练配置和每个预训练阶段的长度混合比例:

| 文件 | 主 context | 长度组成 |
|---|---:|---|
| `stage_B_mix.yaml` | 8K | 全部 8K 主候选 + 4K/16K 受控 warm-up/replay |
| `stage_C1_mix.yaml` | 64K | 全部 64K 主候选 + 4K/8K/16K/32K 受控 replay |
| `stage_C2_mix.yaml` | 128K | 全部 128K 主候选 + 8K/16K/64K 受控 replay |
| `stage_D_mix.yaml` | 256K | 全部 256K 主候选 + 8K/64K/128K 受控 replay |

新增训练配置:

| 文件 | 作用 |
|---|---|
| `model_large.json` | CropGenome-FM-Large 模型结构、token vocabulary、MLM 和后端优先级 |
| `train_stage_B.json` | Stage B 训练步数、batch、学习率、保存和评估间隔 |
| `train_stage_C1.json` | Stage C1 训练配置 |
| `train_stage_C2.json` | Stage C2 训练配置 |
| `train_stage_D.json` | Stage D 训练配置 |
| `zuowu_genomemodel_env.yml` | 已安装训练环境的 mamba 导出文件，含 numpy、CUDA PyTorch、CUDA nvcc、基础构建工具等 |

### `metadata/`

只保留训练解释、split 防泄漏和 QC 追踪需要的小文件:

| 文件 | 作用 |
|---|---|
| `assemblies.canonical.tsv` | 去重后的 canonical assembly 清单 |
| `assembly_splits.canonical.tsv` | canonical accession 级 train/val/test split |
| `assembly_splits.canonical.summary.tsv` | split 统计，确认跨 split 重复为 0 |
| `annotation_qc.summary.tsv` | 注释解析 QC 汇总 |
| `fasta_qc.summary.tsv` | FASTA QC 汇总 |
| `region_candidates.summary.tsv` | 区域候选生成统计 |
| `seqid_alias.summary.tsv` | annotation seqid 到 FASTA contig 的映射统计 |
| `token_vocab.tsv` | input id 编码，A=0 C=1 G=2 T=3 N/ambiguous=4 |

### `inputs/`

训练服务器直接读取的固化输入。每个 stage 包含:

| 文件 | 作用 |
|---|---|
| `manifest.tsv` | shard 级 token 数、窗口数、input/windows SHA256 |
| `summary.tsv` | stage 级写入 token/window 数、过滤失败数、quota 达成情况 |
| `shard_*.input_ids.bin` | uint8 token 序列 |
| `shard_*.windows.tsv.gz` | 每个窗口的来源 assembly、contig、坐标、split、context、region bucket、offset 和 length |

### `scripts/`

训练服务器上的脚本目录保持最小化:

| 文件 | 作用 |
|---|---|
| `check_package.py` | 搬运包和 stage manifest 自检 |
| `train.py` | 正式预训练入口，动态 mask、动态 labels、RC augmentation、checkpoint |
| `run_stage.sh` | 统一启动脚本，默认调用 `mamba run -n zuowu_genomemodel` |
| `requirements.txt` | 训练依赖摘要 |

### 环境说明

`configs/zuowu_genomemodel_env.yml` 来自当前 `zuowu_genomemodel` 环境，核心版本为 `numpy 2.2.6`、`torch 2.5.1`、`torch.version.cuda 12.4`、`cuda-nvcc 12.4.131`。登录节点没有 GPU，因此 `torch.cuda.is_available()` 返回 `False` 属于预期。

已尝试在登录节点用 pip 安装 `mamba-ssm`，但其源码构建强制编译多个 GPU 架构，`cicc` 被系统终止；当前导出环境不包含 `mamba-ssm`，训练脚本会自动使用 `hyena_lite` 后端。若训练服务器有充足编译资源，可在训练服务器上补装 `mamba-ssm` 后启用 Mamba2 后端。

## 已生成 stage 输入

以下为 2026-06-10 23:47:01 CST 根据 `inputs/<Stage>/summary.tsv`、`manifest.tsv` 和 `du` 记录的实际结果。

| stage | 新固化策略 | 实际窗口 | 实际写入 token | 实际目录大小 |
|---|---|---:|---:|---:|
| Stage_B | 全部 8K 主候选 + 4K/16K 受控补充 | 5,594,781 | 41,242,505,216 | 39GB |
| Stage_C1 | 全部 64K 主候选 + 4K/8K/16K/32K 受控补充 | 779,304 | 20,470,165,504 | 20GB |
| Stage_C2 | 全部 128K 主候选 + 8K/16K/64K 受控补充 | 101,293 | 6,898,204,672 | 6.5GB |
| Stage_D | 全部 256K 主候选 + 8K/64K/128K 受控补充 | 18,400 | 2,777,841,664 | 2.6GB |

四个 stage 合计 `71,388,717,056` token，搬运目录约 `67G`。这里的候选窗口不是原始 genome 全量窗口，而是已经经过硬质量过滤、区域采样、去冗余和 split 防泄漏后的 stage 候选池。

## 不在搬运目录中的本地中间文件

以下目录已移出 `training_server_transfer/`，保留在本服务器 `local_intermediate_not_for_transfer/training_server_transfer_legacy/`，训练服务器不需要搬运:

- `annotation_index/`
- `data_manifests/`
- `sampling_index/`
- `sequence_index/`
- 旧 `TRANSFER_MANIFEST.tsv`

## 训练端读取规则

- `.input_ids.bin` 使用 `uint8` 读取。
- 每条窗口的 offset/length 在对应 `.windows.tsv.gz` 中。
- dynamic mask、MLM labels、reverse-complement augmentation、batch shuffle 在训练端在线完成。
- train/val/test split 已按 canonical assembly accession 固定，避免同一 assembly 的窗口泄漏到多个 split。
