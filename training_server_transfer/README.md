# training_server_transfer

这个目录是训练服务器唯一需要搬运的目录。训练服务器不需要访问原始 `/home/user/zhangzhishuai/data/plantDB/genome`，也不需要本服务器生成的中间大索引。

目录结构:

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

训练输入说明:

- `.input_ids.bin`: uint8 token 序列，按窗口顺序连续存储。
- `.windows.tsv.gz`: 每个窗口的来源 assembly、contig、坐标、split、context、region bucket、shard offset 和 length。
- 每个 stage 的 `manifest.tsv`: shard 级 token 数、窗口数、input/windows SHA256。
- 每个 stage 的 `summary.tsv`: 实际写入 token/window 数、过滤失败数、quota 达成情况。
- `metadata/token_vocab.tsv`: token 编码，A=0, C=1, G=2, T=3, N/ambiguous=4。

训练端注意:

- dynamic mask、reverse-complement augmentation、batch shuffle 在训练服务器在线完成；这里没有固化 mask/labels。
- split 已按 canonical assembly accession 固定，避免同一 assembly 的窗口同时进入 train/val/test。
- Stage 顺序建议为 B -> C1 -> C2 -> D；也可以只搬运当前要训练的 stage 子目录以减少训练服务器占用。
- 搬运后先运行 `sha256sum -c SHA256SUMS`，再启动训练。
