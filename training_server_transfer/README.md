# training_server_transfer

这个目录是本项目专门用于跨服务器训练的数据传输目录。

用户把这个目录中的全部内容传输到训练服务器后，训练服务器应能读取当前 stage 的固化输入、必要索引和配置，并开始训练。GitHub 只保留本说明文件；实际生成的大文件会被 `.gitignore` 排除，不上传到仓库。

正式传输目录结构:

```text
training_server_transfer/
  README.md
  TRANSFER_MANIFEST.tsv
  configs/
  data_manifests/
  sequence_index/
  annotation_index/
  sampling_index/
  stage_inputs/
    Stage_B/
      manifest.tsv
      stage_mix.yaml
      shard_000001.input_ids.bin
      shard_000001.windows.tsv
      shard_000001.sha256
      ...
```

每次推荐只放当前要训练的一个 stage，例如先放 `Stage_B`。Stage B 训练完成后，删除或归档 `stage_inputs/Stage_B/`，再重新准备包含 `Stage_C1` 的传输目录。

本目录应包含:

- 当前 stage 固化输入窗口或 `input_ids`。
- 当前 stage 的 `manifest.tsv`。
- 当前 stage 的 `stage_mix.yaml`。
- 必要的 data/sequence/annotation/sampling 小索引。
- 训练配置。
- sha256 校验文件。

本目录不应包含:

- 固定 mask 位置。
- 固定 MLM labels。
- 固定 batch 顺序。
- 固定 RC 增强结果。
- 历史 checkpoint。
- 下游大结果文件。
- 原始 plantDB 全量数据，除非后续明确改为训练服务器重新取序列。

训练服务器收到本目录后必须先检查:

- `TRANSFER_MANIFEST.tsv` 文件列表完整。
- 所有 shard 的 sha256 校验通过。
- stage mix 比例符合计划。
- split 信息没有 train/val/test 泄漏。
- 当前 stage 输入体积和训练服务器剩余磁盘空间满足要求。
