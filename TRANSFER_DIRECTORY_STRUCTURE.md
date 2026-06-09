# training_server_transfer 目录结构和搬运说明

更新时间: 2026-06-09 12:52:00 CST

## 1. 目标

`training_server_transfer/` 是本项目唯一推荐整体搬运到训练服务器的目录。训练服务器不能访问本服务器的原始 genome 目录，因此最终训练前必须把训练所需的索引、配置、split、候选区域和 Stage B/C1/C2/D 固化输入都放入这个目录。

用户搬运时只需要搬运:

```text
training_server_transfer/
```

到训练服务器后，先校验:

```bash
cd training_server_transfer
sha256sum -c SHA256SUMS
```

校验通过后，训练程序只读取这个目录，不再访问 `/home/user/zhangzhishuai/data/plantDB/genome`。

## 2. 当前目录状态

当前已完成的是基础索引包，体积约 `5.3GB`。它已经可以搬运到训练服务器用于检查、开发 dataloader、做 dry-run 配置，但还不能直接启动正式预训练，因为 Stage B/C1/C2/D 的固化 `input_ids` 尚未全部生成。

当前已完成:

- crop manifest: 263 个 crop assembly，26 个属。
- FASTA QC: 263/263。
- annotation QC: 263/263。
- contig index。
- region candidates。
- assembly split。
- stage mix 配置。
- transfer manifest。
- SHA256 校验。

仍需完成:

- 处理 annotation seqid 与 FASTA contig id 不匹配问题。
- 按 hard filter 和区域保留比例生成最终 window candidates。
- 按 Stage B/C1/C2/D 比例固化 `input_ids` shards。
- 为每个 stage 生成 shard manifest 和 SHA256。

## 3. 最终目录结构

最终训练前，`training_server_transfer/` 应包含:

```text
training_server_transfer/
  README.md
  TRANSFER_MANIFEST.tsv
  SHA256SUMS

  configs/
    stage_B_mix.yaml
    stage_C1_mix.yaml
    stage_C2_mix.yaml
    stage_D_mix.yaml
    model_large.yaml
    train_stage_B.yaml
    train_stage_C1.yaml
    train_stage_C2.yaml
    train_stage_D.yaml

  data_manifests/
    assemblies.tsv
    assemblies.summary.txt
    assembly_splits.tsv
    assembly_splits.summary.tsv

  sequence_index/
    contigs.tsv
    fasta_qc.summary.tsv
    seqid_alias.tsv

  annotation_index/
    annotation_qc.summary.tsv

  sampling_index/
    region_candidates.tsv.gz
    region_candidates.summary.tsv
    final_window_candidates.tsv.gz
    final_window_candidates.summary.tsv

  stage_inputs/
    Stage_B/
      manifest.tsv
      stage_mix.yaml
      shard_000001.input_ids.bin
      shard_000001.windows.tsv
      shard_000001.sha256
      ...
    Stage_C1/
      manifest.tsv
      stage_mix.yaml
      shard_000001.input_ids.bin
      shard_000001.windows.tsv
      shard_000001.sha256
      ...
    Stage_C2/
      manifest.tsv
      stage_mix.yaml
      shard_000001.input_ids.bin
      shard_000001.windows.tsv
      shard_000001.sha256
      ...
    Stage_D/
      manifest.tsv
      stage_mix.yaml
      shard_000001.input_ids.bin
      shard_000001.windows.tsv
      shard_000001.sha256
      ...
```

## 4. 顶层文件

### `README.md`

说明 `training_server_transfer/` 的用途、目录结构、校验方式和注意事项。这个文件可以上传 GitHub。

### `TRANSFER_MANIFEST.tsv`

记录需要搬运的所有文件。字段:

| 字段 | 含义 |
|---|---|
| `relative_path` | 文件在 `training_server_transfer/` 内的相对路径 |
| `absolute_path` | 本服务器上的源路径或生成路径 |
| `bytes` | 文件大小 |
| `role` | 文件角色 |
| `required` | 是否训练必须 |

### `SHA256SUMS`

所有可搬运文件的 SHA256 校验表。搬到训练服务器后必须运行:

```bash
sha256sum -c SHA256SUMS
```

如果有任何文件校验失败，不能开始训练。

## 5. `configs/`

### `stage_B_mix.yaml`

Stage B 的长度比例配置:

```text
70% 8K + 20% 4K + 10% 16K
```

### `stage_C1_mix.yaml`

Stage C1 的长度比例配置:

```text
70% 64K + 15% 8K + 10% 16K/32K + 5% 4K
```

### `stage_C2_mix.yaml`

Stage C2 的长度比例配置:

```text
75% 128K + 15% 64K + 10% 8K/16K
```

### `stage_D_mix.yaml`

Stage D 的长度比例配置:

```text
80% 256K + 15% 128K + 5% 8K/64K
```

### `model_large.yaml`

正式 Large 模型结构配置。后续生成，包含 layers、hidden size、attention interval、loss 权重、RC consistency 等。

### `train_stage_*.yaml`

每个 stage 的训练配置。后续生成，包含 batch 组织、gradient accumulation、checkpoint 策略、日志路径和评估间隔。

## 6. `data_manifests/`

### `assemblies.tsv`

263 个 crop assembly 的总清单。包含 assembly id、物种、属、genome 路径、annotation 路径、assembly level、source 等。训练服务器不使用原始路径读取 genome，但需要这些字段用于 metadata 和结果解释。

### `assemblies.summary.txt`

assembly 数量、属数量、压缩 genome/annotation 体积摘要。

### `assembly_splits.tsv`

严格 assembly-level split。当前:

| split | assembly 数 |
|---|---:|
| train | 197 |
| val | 35 |
| test | 31 |

训练、验证、测试不能跨 assembly 混用，避免同一基因组片段泄漏。

### `assembly_splits.summary.tsv`

split 数量摘要。

## 7. `sequence_index/`

### `contigs.tsv`

FASTA QC 结果。每条 contig/chromosome 一行，包含:

- assembly id。
- species/genus。
- contig id。
- length。
- A/C/G/T/N 计数。
- GC fraction。
- N fraction。
- softmask fraction。
- max N run。
- organelle 候选标记。
- 是否训练可用。

### `fasta_qc.summary.tsv`

每个 assembly 的 FASTA QC 摘要，包括 contig 数、总长度、低质量 contig 数和状态。

### `seqid_alias.tsv`

后续生成。用于解决 annotation seqid 与 FASTA contig id 不一致问题。训练窗口生成必须使用这个表把 annotation 坐标映射到真实 FASTA contig。

## 8. `annotation_index/`

### `annotation_qc.summary.tsv`

每个 assembly 的注释解析摘要，包括 gene、mRNA/transcript、exon、CDS、UTR 数量，以及 bad coordinate、parent missing、CDS phase 异常等。

### 不搬运 `features.tsv`

`annotation_index/features.tsv` 约 24GB，是本服务器构建候选区间的中间文件。它不进入最终搬运目录，避免训练服务器存储压力过大。训练服务器只需要已经生成好的 candidates 和 stage inputs。

## 9. `sampling_index/`

### `region_candidates.tsv.gz`

功能区域候选区间。由 GFF/GTF 注释和 contig QC 构建，包含:

- CDS。
- exon。
- splice flank。
- UTR。
- gene body。
- promoter core 0-5kb。
- promoter distal 5-20kb。
- TES flank。

当前文件约 `3.2GB`。

### `region_candidates.summary.tsv`

候选区域数量统计。当前已发现一部分 annotation seqid 没有匹配到 FASTA contig id，这部分会在 `seqid_alias.tsv` 生成和候选重建后修正。

### `final_window_candidates.tsv.gz`

后续生成。它是正式 stage input 生成前的窗口候选池，已经完成:

- hard quality filter。
- 区域保留比例。
- 去冗余。
- split 标记。
- context bucket 标记。
- region bucket 标记。

### `final_window_candidates.summary.tsv`

后续生成。统计每个 split、stage、context bucket、region bucket 的窗口数量和 token 数。

## 10. `stage_inputs/`

这是最终训练服务器直接读取的核心数据目录。训练服务器不能访问原始 genome，因此这里必须包含已经固化好的输入。

### `Stage_B/`

Stage B 输入。目标比例:

```text
70% 8K + 20% 4K + 10% 16K
```

文件:

| 文件 | 含义 |
|---|---|
| `manifest.tsv` | Stage B 所有 shard 的清单、token 数、region/context 比例和 SHA256 |
| `stage_mix.yaml` | Stage B 长度比例配置副本 |
| `shard_*.input_ids.bin` | 固化的 `uint8 input_ids`，A/C/G/T/N 等已经编码 |
| `shard_*.windows.tsv` | 每条 window 的坐标、assembly、contig、split、region bucket、context bucket |
| `shard_*.sha256` | 单个 shard 的校验 |

训练时仍然动态生成:

- mask positions。
- MLM labels。
- RC augmentation。
- batch order。
- dynamic loss 权重微调。

### `Stage_C1/`

Stage C1 输入。目标比例:

```text
70% 64K + 15% 8K + 10% 16K/32K + 5% 4K
```

文件结构同 Stage B。

### `Stage_C2/`

Stage C2 输入。目标比例:

```text
75% 128K + 15% 64K + 10% 8K/16K
```

文件结构同 Stage B。

### `Stage_D/`

Stage D 输入。目标比例:

```text
80% 256K + 15% 128K + 5% 8K/64K
```

文件结构同 Stage B。Stage D 是资源允许后的长上下文 midtraining，不是第一版必须完成条件。

## 11. 搬运规则

最终只搬运:

```text
training_server_transfer/
```

不要单独搬运:

- `/home/user/zhangzhishuai/data/plantDB/genome`
- `annotation_index/features.tsv`
- `logs/`
- `checkpoints/`
- `results/`
- `scripts/`
- `slurm/`

训练服务器收到后:

1. 进入目录。
2. 运行 `sha256sum -c SHA256SUMS`。
3. 检查 `TRANSFER_MANIFEST.tsv`。
4. 检查 `stage_inputs/Stage_B/manifest.tsv`。
5. 启动 Stage B 训练。

## 12. 当前下一步

当前下一步不是搬运，而是在本服务器继续完成:

1. 生成 `seqid_alias.tsv`。
2. 重建修正后的 `region_candidates.tsv.gz`。
3. 生成 `final_window_candidates.tsv.gz`。
4. 生成 Stage B/C1/C2/D 的 `input_ids` shards。
5. 更新 `TRANSFER_MANIFEST.tsv` 和 `SHA256SUMS`。

以上完成后，`training_server_transfer/` 才是“搬过去即可直接训练”的最终目录。
