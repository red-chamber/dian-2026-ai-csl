# checkpoints

训练产出的模型权重放这里。整个目录在 `.gitignore` 里被忽略，只保留本文件，所以克隆仓库后这个目录是空的（需要时按下面的命令重新训练生成）。

## 命名约定

| 文件名 | 内容 | 用途 |
| --- | --- | --- |
| `<tag>_best.pt` | 验证指标最好的那一轮的模型权重 | 交付与推理用 |
| `<tag>_last.pt` | 最近一轮的权重，另外带优化器状态与 epoch | 断点续训用，只有 Level 4 会产生 |

`<tag>` 是训练时 `--tag` 指定的名字，默认值与实验编号对应，同名记录可以在 `reports/metrics/<tag>.json` 里找到：

| tag | 对应实验 |
| --- | --- |
| `mlp_mnist` | 实验 01，MLP on MNIST |
| `cnn_mnist` | 实验 02，CNN on MNIST |
| `alexnet_fashionmnist` / `resnet_fashionmnist` / `resnet_plain` | 实验 03，经典网络与残差消融 |
| `unet_l1` / `unet_mse` / `unet_l1grad` | 实验 04 与 05，U-Net 的三种损失对照 |

## 权重里存了什么

`torch.save` 保存的是一个字典：

- `model_state`：`model.state_dict()`，全部参数张量
- `model_config`：重建模型所需的结构参数（例如 U-Net 的 `base_channels` / `depth`）。加载时先用它把模型搭出来再灌权重，所以推理脚本不用手写网络结构
- `epoch`：保存时是第几轮
- `val_acc`（Level 1–3）或 `val_psnr` / `val_ssim`（Level 4）：当时的验证指标
- `args`：训练时的命令行配置，用于追溯超参

为什么只存 `state_dict` 而不是整个模型对象：存整个对象会把类的定义一起序列化进去，代码结构一改或者换了环境就可能加载不了。存 `state_dict` 只要求模型结构和权重对得上，配合 `model_config` 就能重建，更稳。

## 怎么重新生成

```bash
python level1_mlp/src/train.py
python level2_cnn/src/train.py
python level3_classic_networks/src/train.py --model alexnet
python level3_classic_networks/src/train.py --model resnet
python level4_unet/src/train.py --data-root /path/to/dataset
```

完整超参与命令见根 README 的「运行训练」，各 Level 的结构细节见对应目录的 README。

## 体积

Level 4 的 `unet_l1_best.pt` 有 31,036,481 个参数，按 float32 算约 124 MB。Level 1–3 小一些但同样不适合入库（ResNet-18 也有 1100 万参数）。这也是 `.gitignore` 整体忽略这个目录的原因。
