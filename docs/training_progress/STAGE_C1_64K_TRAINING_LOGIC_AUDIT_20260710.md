# Stage C1 64K 训练逻辑审计与修复闭环

更新时间：2026-07-10 19:03 CST

## 结论

首次 Stage C1 运行虽然能接收 64K tensor（张量），但存在两个 P0 问题：token 级有效依赖只有约 8K，且 64K 窗口仅获得约 31.10% 的优化权重。该运行已安全停止并仅保留为 diagnostic（诊断记录），不作为正式训练结果。

两项科学语义问题和容错问题均已修复。修复后的实现通过四项正式 gate：

1. **Execution gate（执行门）**：A100 GPU2 完成真实 65,536 bp 前向、反向、梯度裁剪和 optimizer step（优化器更新）。
2. **Dependency gate（依赖门）**：保持 128 个 chunk 的等拓扑梯度实验由旧结构局部 992 个位置扩展到 8192/8192 全长位置；按 chunk size 从 64 扩到真实 512，对应完整 65,536 bp 依赖拓扑。
3. **Objective gate（目标门）**：MLM、RC 和 region 三类目标分别按有效 token/标签数归一化，不再让 4K 与 64K 窗口等权。
4. **Selection gate（模型选择门）**：固定验证面板使用确定性无放回顺序，从 32 扩到 256 windows，覆盖 22 assemblies、11 species、7 个 region 类别和 76 个 64K 窗口。

corrected Stage C1 正式训练已从锁定的 step14000 在 A100 GPU2 启动。本文证明工程和训练语义已满足正式长上下文训练前提，但**不证明 64K 已经比 8K 的下游任务更准**；该结论必须等待独立长程 benchmark（基准评测）。

## 1. 原运行为什么不能保留为正式结果

### 1.1 P0：64K 输入不等于 64K token 级依赖

原结构由 kernel=127 的本地 `HyenaLiteBlock` 和固定 512-bp 非重叠分块注意力组成。梯度支持实验显示，16K 输入的中心输出只能依赖约 8,184 bp；区间外梯度精确为 0。也就是说，模型虽然能装入 64K 输入，但单个 MLM 预测看不到完整 64K。

### 1.2 P0：变长窗口按窗口而不是按 token 加权

原循环先对每个 micro-batch 内的 token loss 取均值，再把 128 个 micro-batch 等权累积。micro-batch size=1 时，一个 4K 窗口与一个 64K 窗口近似等权。

| bucket | 数据 token 比例 | 原等窗口优化权重 |
|---|---:|---:|
| 4K | 约 3.7% | 23.96% |
| 8K | 约 11.2% | 35.95% |
| 16K/32K | 约 7.5% | 8.99% |
| 64K | 约 77% | 31.10% |

因此原运行明显高估短窗口贡献，不能解释为预设的长窗口主导课程。

### 1.3 P1：选择与恢复面不足

- 原验证每次只取 32/7,576 个 val 窗口，并允许重复抽样；只适合 health check（健康检查），不足以单独选择最长 180k-step 运行的 checkpoint。
- checkpoint 直接写最终文件，写入中断可能破坏旧文件。
- warmstart（跨阶段只继承模型）与 exact resume（同阶段恢复）没有显式分离。
- 没有 non-finite loss/gradient guard（非有限损失/梯度保护）、终止信号保存和重复启动锁。
- 非交互 SSH 环境可能找不到 mamba 并错误回退到系统 Python；首次修复 gate 就真实暴露了该问题。

## 2. 修复内容

### 2.1 无新增参数的 dilated chunk attention

Stage C1 使用独立配置 `configs/model_stage_C1_64k.json`。8 个注意力层的 dilation 为：

```text
1, 2, 4, 8, 16, 32, 64, 128
```

实现只改变 token 重排和连接关系，不新增参数；step14000 的 430 个 state-dict keys（权重键）严格加载，missing/unexpected/shape mismatch 全部为 0。

scaled dependency probe（缩放依赖实验）保持真实结构的 128 个 chunk，只把 chunk size 从 512 缩到 64：

| 结构 | 非零梯度支持 |
|---|---:|
| 旧固定分块 | 位置 3632–4623，共 992/8192 |
| 新膨胀分块 | 位置 0–8191，共 8192/8192 |

### 2.2 按目标自身分母精确归一化

- MLM：按非 ignore 的 masked token 数量。
- RC consistency（正反链一致性）：按有效序列 token 数量。
- Region classification（区域分类）：按有效区域标签数量。
- 每个 gradient-accumulation window（梯度累计窗口）先统计全局分母，再对每个 micro-batch 的分子加权反传。
- 指标汇总同样使用 token/标签分母，不再平均 batch mean（批均值）。

专项人工梯度验证：期望 `2.8823529412`，实际 `2.8823530674`，差异仅为 float32 浮点误差。

### 2.3 固定无放回验证面板

`eval_batches` 从 32 增至 256；验证迭代使用固定 seed 的无放回排列。当前固定面板实际覆盖：

| 项目 | 覆盖 |
|---|---:|
| assemblies | 22 |
| species | 11 |
| region classes | 7/7 |
| 4K / 8K / 16K / 32K / 64K | 65 / 89 / 17 / 9 / 76 |

### 2.4 恢复、停止与路径安全

- checkpoint 写入同目录唯一 `.tmp`，成功后 `os.replace` 原子替换；失败时清理临时文件并保留旧 checkpoint。
- `warmstart`：严格继承模型权重，重置 optimizer、global step 和 best tracking。
- `exact`：同 Stage 恢复模型、optimizer、global step 和 early-stop tracking；它表示训练状态恢复，不宣称跨 DataLoader worker 的逐 bit 重放。
- 只有 `optimizer.step()` 成功后才推进 `last_step`。
- 更新前检查 loss component 和 gradient norm 是否 finite（有限）。
- SIGTERM/SIGINT 只设置停止标志；完成当前 optimizer-step 边界后原子保存 `interrupted_step_*.pt`。launcher 会把信号转发给训练子进程。
- live PID + 原子 lock directory 防止重复启动，死 PID 的 stale lock（陈旧锁）可恢复。
- 常规 checkpoint 从每 3000 step 提前到每 500 step；跑到 step12000 约 24 个 checkpoint、约 106GB，共享盘可承受。
- `safe_relative_path('.')` 现已拒绝，避免把阶段输出错误写到 package root（包根目录）。

### 2.5 非交互解释器 provenance

launcher 的解释器顺序为：

1. `PYTHON_BIN_OVERRIDE`；
2. `CONDA_ENV_PREFIX/bin/python`；
3. 可用的 mamba 环境；
4. `$HOME/.local/share/mamba/envs/<name>/bin/python`；
5. 系统 Python，但仍必须通过依赖检查。

GPU 前 preflight 会打印 `sys.executable`、NumPy/PyTorch/CUDA 版本并完成 import。正式 gate 使用：

```text
python=/home/user/zhangzhishuai/.local/share/mamba/envs/zuowu_genomemodel/bin/python
numpy=2.2.6, torch=2.5.1, torch_cuda=12.4, cuda_available=true
```

## 3. 最终验证证据

### 3.1 CPU 与静态检查

- `training_server_transfer.tests.test_train_v2`：33/33 通过。
- `py_compile`：通过。
- `bash -n scripts/run_stage.sh`：通过。
- `git diff --check`：通过。
- 新旧模型 state-dict keys 严格兼容：通过。
- fresh scaled dependency probe 和 mixed-length gradient calculation：通过。

### 3.2 A100 GPU2 真实 64K gate

| 项目 | 结果 |
|---|---:|
| 输入 | `[1, 65536]` |
| checkpoint | early-stop step14000；SHA-256=`c81bce39...c83fed` |
| 权重加载 | 430 keys；missing/unexpected/mismatch=`0/0/0` |
| 总 / MLM / RC / region loss | `0.717403 / 0.586707 / 0.363610 / 2.468471` |
| selection loss | `0.593979` |
| allocated / reserved 显存 | `26,416.9 / 27,946.0 MiB` |
| forward/backward/gradient clip/optimizer step | PASS |

## 4. 正式重启快照

- 启动基座：`checkpoint_stage_B_8k_final.pt → checkpoint_best.pt = step14000`。
- 初始化模式：`warmstart`；optimizer/global step/best tracking 从 Stage C1 的 0 开始。
- GPU：A100 physical GPU2，`CUDA_VISIBLE_DEVICES=2`。
- launcher PID：54388。
- training PID：55762。
- 日志：`training_server_transfer/runs/Stage_C1/train_step14000_64k_corrected.log`（本地运行产物，不上传 GitHub）。
- 启动检查：430/430 keys 严格加载；best tracking 来源为 fresh；GPU 利用率观测到 87%，训练进程持续运行。
- 首个日志进度点：step10；首个常规原子 checkpoint：step500。

第一次诊断运行 PID13652 已安全停止，其日志 `train_step14000_64k.log` 仅保留本地用于审计，不参与正式结果。

## 5. 结论边界

可以说：修复后的 Stage C1 满足正式 64K 训练的执行、远程依赖、目标归一化、模型选择和恢复条件。

不能说：64K 已经在生物学下游任务上优于 8K。后续必须使用预先定义的 validation-only checkpoint 候选和长程任务验证，不能在每个 checkpoint 上反复触碰 formal test（正式测试集）。