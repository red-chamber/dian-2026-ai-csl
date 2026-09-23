# 关于 configs/ 目录

这个目录暂时不放配置文件，原因和用法记在这里。

## 为什么用命令行参数而不是 YAML 配置

根 README 的结构图里声明了 `configs/`，但 Level 4 最终选择了命令行参数
（与 Level 1-3 一致）。两个理由：

1. **一致性**。四个 Level 的入口脚本都是 `argparse`，看一眼就知道怎么改参数，
   不用再学一套配置文件的写法。
2. **配置快照自动留存**。`train.py` 会把本次生效的全部超参写进
   `reports/metrics/<tag>.json`（含 `crop_size`、`loss_fn`、`lr`、`scheduler`、
   `train_time_hour`、`estimated_cost` 等），本身就是一份比 YAML 更完整的实验记录 ——
   它连实测结果一起存了。所以「可复现的配置」这件事已经由指标文件承担了。

需要几组对照实验时，把命令写成一个 shell 脚本即可，见下面。

## 常用的命令组

```bash
# 数据集的路径，按自己的环境改
DATA=/path/to/dataset

# --- 主实验：L1 损失，基准配置 ---
python level4_unet/src/train.py --data-root $DATA --loss l1 --tag unet_l1

# --- 损失函数对照 ---
python level4_unet/src/train.py --data-root $DATA --loss mse     --tag unet_mse
python level4_unet/src/train.py --data-root $DATA --loss l1_grad --tag unet_l1grad

# --- 显存不足时（本机 8 GB） ---
python level4_unet/src/train.py --data-root $DATA --crop 256 --batch-size 8 \
    --base-channels 32 --tile 512 --tag unet_small

# --- 云上训练：续训 + 费用估算 ---
python level4_unet/src/train.py --data-root $DATA --epochs 60 \
    --tag unet_l1 --resume --gpu-hourly-cost 1.5

# --- 训练完统一评估 ---
python level4_unet/src/evaluate.py --checkpoint checkpoints/unet_l1_best.pt --worst 4
python level4_unet/src/evaluate.py --checkpoint checkpoints/unet_mse_best.pt --worst 4
python level4_unet/src/evaluate.py --checkpoint checkpoints/unet_l1grad_best.pt --worst 4
```

## 产物去哪了

| 产物 | 路径 |
|---|---|
| 最优权重 | `checkpoints/<tag>_best.pt` |
| 最近一轮 + 优化器状态 | `checkpoints/<tag>_last.pt` |
| 训练指标与全部超参 | `reports/metrics/<tag>.json` |
| 测试集指标 | `reports/metrics/<tag>_test.json` |
| 曲线与对比图 | `reports/figures/<tag>_*.png` |
| 推理结果 | `reports/samples/` |

`checkpoints/` 与 `data/` 都在 `.gitignore` 里，不会被提交 —— 权重文件动辄上百 MB，
数据集更不该进仓库。
