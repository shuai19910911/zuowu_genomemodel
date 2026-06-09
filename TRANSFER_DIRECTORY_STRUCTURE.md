# training_server_transfer 目录结构和搬运说明

更新时间: 2026-06-09 21:10:00 CST

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

当前最终目录约 `50GB`，全目录 SHA256 校验已于 2026-06-09 通过。

## 简约目录结构

```text
training_server_transfer/
  README.md
  MANIFEST.tsv
  SHA256SUMS
  configs/
    stage_B_mix.yaml
    stage_C1_mix.yaml
    stage_C2_mix.yaml
    stage_D_mix.yaml
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
```

## 目录说明

### `configs/`

保存每个预训练阶段的长度混合比例:

| 文件 | 主 context | 长度组成 |
|---|---:|---|
| `stage_B_mix.yaml` | 8K | 70% 8K + 20% 4K + 10% 16K |
| `stage_C1_mix.yaml` | 64K | 70% 64K + 15% 8K + 10% 16K/32K + 5% 4K |
| `stage_C2_mix.yaml` | 128K | 75% 128K + 15% 64K + 10% 8K/16K |
| `stage_D_mix.yaml` | 256K | 80% 256K + 15% 128K + 5% 8K/64K |

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

## 已生成 stage 输入

| stage | 写入 token | 窗口数 | shard 数 | 目录大小 |
|---|---:|---:|---:|---:|
| Stage_B | 30,600,306,688 | 4,295,686 | 31 | 29GB |
| Stage_C1 | 15,301,525,504 | 700,433 | 16 | 15GB |
| Stage_C2 | 5,102,608,384 | 87,591 | 6 | 4.8GB |
| Stage_D | 2,045,698,048 | 15,612 | 3 | 2.0GB |

质量过滤结果:

| stage | failed_quality | missing_contig_in_fasta |
|---|---:|---:|
| Stage_B | 1,513 | 0 |
| Stage_C1 | 254 | 0 |
| Stage_C2 | 32 | 0 |
| Stage_D | 4 | 0 |

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
