# Formal CaduceusRC Stage_B stopped-run evaluation

更新时间: 2026-06-22 12:30 CST

## 1. 停止状态

用户要求停止当前训练后，已精确终止 `formal_caduceus_rc_stageB_mb5` 训练进程树：`mamba run` 父进程、主 `python scripts/train.py` 进程和 5 个 DataLoader worker（数据加载子进程）均已退出。GPU2（2号显卡）显存已释放到 0 MiB（显存占用）。

| 项目 | 数值 |
|---|---:|
| 停止时最后 train step（训练步） | 8500 |
| 最新完整 checkpoint（模型存档点） | step8000 |
| 最新 train loss（训练损失） | 1.037332 |
| 最新 MLM loss（masked language modeling，遮盖碱基预测损失） | 1.034064 |
| 最新 RC loss（reverse-complement consistency，反向互补一致性损失） | 0.108925 |
| 最新 learning rate（学习率） | 1.70e-04 |
| 最新 validation step（验证步） | 8000 |
| 最新 validation loss（验证损失） | 1.122568 |
| 最佳 validation step（验证步） | 5000 |
| 最佳 validation loss（验证损失） | 1.079774 |

停止时 step8500 尚未形成 checkpoint（模型存档点），所以可复用的最新权重文件仍是 `training_server_transfer/runs/Stage_B_formal_caduceus_rc/checkpoints/step_00008000.pt`。

## 2. 训练曲线评价

训练本身是有效的：train loss（训练损失）从 1.641037 降到 1.037332，说明模型确实学到了 Stage_B（第二阶段）预训练任务中的序列统计规律。MLM loss（遮盖碱基预测损失）同步下降，是总 loss（总损失）下降的主因。

validation loss（验证损失）不是单调下降：

| validation step（验证步） | val loss（验证损失） | val MLM loss（验证遮盖碱基预测损失） | val RC loss（验证反向互补一致性损失） |
|---:|---:|---:|---:|
| 1000 | 1.244846 | 1.244806 | 0.001336 |
| 2000 | 1.235688 | 1.235389 | 0.009966 |
| 5000 | 1.079774 | 1.076772 | 0.100065 |
| 6000 | 1.167987 | 1.166388 | 0.053305 |
| 7000 | 1.088392 | 1.085457 | 0.097815 |
| 8000 | 1.122568 | 1.119921 | 0.088235 |

解释：step5000 是 validation loss（验证损失）最低点；step7000 次好；step8000 比 step5000 差。继续训练到 step8500 时 train loss（训练损失）仍低，但没有新的 validation（验证）和 checkpoint（模型存档点）。因此从“验证集泛化”角度，step5000 更像当前最佳 checkpoint（模型存档点）。

## 3. 下游 probe（探针评测）评价

128 bp（128 个碱基对）region_bucket_classification（功能区域桶分类）first-pass probe（第一轮探针评测）结果如下：

| checkpoint（模型存档点） | Accuracy（准确率） | Macro-F1（类别平均 F1） | Balanced accuracy（类别平均召回） | Delta Macro-F1 vs baseline（相对基线） |
|---|---:|---:|---:|---:|
| step1000 | 0.214286 | 0.200834 | 0.214286 | +0.053405 |
| step2000 | 0.125000 | 0.088710 | 0.125000 | -0.058719 |
| step3000 | 0.214286 | 0.176644 | 0.214286 | +0.029215 |
| step5000 | 0.267857 | 0.242750 | 0.267857 | +0.095321 |
| step8000 | 0.232143 | 0.208736 | 0.232143 | +0.061307 |

baseline（基线）是 1-mer composition（单碱基组成）nearest-centroid（最近质心）模型，Macro-F1（类别平均 F1）为 0.147429。

结论：step5000 是当前最好的 downstream probe（下游探针评测）checkpoint（模型存档点）；step8000 仍明显高于 baseline（基线），但低于 step5000。也就是说，模型学到了有用的区域表示，但继续训练到 step8000 后没有在这个小样本 probe（探针评测）上继续提升。

## 4. 总体判断

这次 Stage_B（第二阶段训练）不是失败：训练 loss（损失）下降、validation loss（验证损失）明显优于早期 step1000/2000，下游 probe（探针评测）也超过简单 baseline（基线）。

但它也还不是可以直接宣称成功的最终大模型：

1. validation loss（验证损失）最佳点在 step5000，不是最新 step8000。
2. downstream probe（下游探针评测）最高点也是 step5000，step8000 回落。
3. 当前 probe（探针评测）每类只有 8 个测试窗口，统计方差很大。
4. 当前 probe（探针评测）只用 128 bp（碱基对），没有验证 512 bp、1 kb（1000 个碱基对）或完整 8 kb（8000 个碱基对）上下文能力。

保守结论：当前模型已经有可迁移表示信号，step5000 是优先候选 checkpoint（模型存档点），step8000 可作为“训练更久但下游回落”的对照 checkpoint（模型存档点）。

## 5. 建议下一步

1. 暂时不要继续盲目训练同一配置。
2. 先把 step5000 和 step8000 做更稳的 downstream benchmark（下游基准评测）：扩大每类窗口数，并至少加入 512 bp（碱基对）或 1 kb（1000 个碱基对）长度。
3. 如果正式 benchmark（基准评测）也显示 step5000 更好，则后续训练应考虑 early stopping（早停）或降低后期 learning rate（学习率），而不是直接跑更长。
4. 如果更大样本 benchmark（基准评测）显示 step8000 更稳，则说明当前 128 bp probe（探针评测）波动太大，应把判断权交给正式 benchmark（基准评测）。
5. 新训练重启前，建议先做学习率/RC loss 权重/采样策略的小规模 ablation（消融实验），再启动正式长跑。

## 6. 关联文件

- Loss curve PNG（损失曲线位图预览）: [`training_curves/formal_caduceus_rc_stageB/figures/formal_caduceus_rc_stageB_loss_curve.png`](training_curves/formal_caduceus_rc_stageB/figures/formal_caduceus_rc_stageB_loss_curve.png)
- Loss curve source data（损失曲线源数据）: [`training_curves/formal_caduceus_rc_stageB/source_data/formal_caduceus_rc_stageB_loss_curve.tsv`](training_curves/formal_caduceus_rc_stageB/source_data/formal_caduceus_rc_stageB_loss_curve.tsv)
- Loss summary source data（损失摘要源数据）: [`training_curves/formal_caduceus_rc_stageB/source_data/formal_caduceus_rc_stageB_loss_summary.tsv`](training_curves/formal_caduceus_rc_stageB/source_data/formal_caduceus_rc_stageB_loss_summary.tsv)
- 128 bp checkpoint trend（训练步趋势）: [`downstream/comparisons/formal_caduceus_rc_128bp_step_trend/`](downstream/comparisons/formal_caduceus_rc_128bp_step_trend/)
