# Level 4：基于 U-Net 的手写笔记擦除

目标：输入带手写内容的试卷 / 作业图片，输出擦除手写内容后的干净图片，保留印刷文字、表格与题目结构

验收标准：
1. 输出 PSNR / SSIM 演化曲线
2. 结果不出现全白 / 全黑
3. 记录训练时长与费用

## 数据集

按日期分三批，每批的 input 是带手写版、output 是同一张图的干净版，两边文件名一一对应：

```
<data_root>/
├── 20250211/dataset/{input,output}/     812 对
├── 20250212/dataset/{input,output}/     900 对
└── 20250213/dataset/{input,output}/     700 对
                                        合计 2412 对
```

实测发现的三个数据集相关的情况：

| 情况 | 处理方式 |
| --- | --- |
| input 格式不统一：部分文件扩展名是 .jpg 但内容其实是 PNG，且带 alpha 通道；另一部分是普通三通道 JPEG。output 则统一是单通道灰度 JPEG | 统一转灰度。带 alpha 的先合成到白底再转灰度 |
| 尺寸差异极大：长边从 143 px 到 2000 px，长宽比各异 | 训练用原分辨率随机裁剪；验证测试用整图 |
| 尺寸不是 16 的整数倍 | U-Net 每级下采样边长减半，输入必须是2^depth 的整数倍。推理前把图补齐到整数倍、跑完再裁回原尺寸 |

**不能靠缩放**凑尺寸。这个任务要求输出与输入逐像素对齐，缩放会破坏这个对应关系，擦除结果贴不回原图。

---

## 数据划分：按日期留出

```
train   20250211 + 20250212                     1712 对
val     20250213 的前 10%                         70 对
test    20250213 其余                            630 对
```

---

## 为什么用原分辨率裁剪而不是整图缩放

把 2000 px 的长边缩到 256，手写的细笔迹会和印刷字糊在一起，任务难度大大增加，所以不能缩放。
- 训练：从原图随机裁 384×384（或 256），不缩放
- 推理：整图送入（U-Net 是全卷积结构，不依赖固定输入尺寸）

原分辨率裁剪，还有一个好处，即相当于数据增强，部分弥补了数据量小的缺点

## U-Net

```
输入 (1, H, W)
  ├─ enc1: ConvBlock(1   -> 64)   ─────────────────────┐ 跳跃连接
  │   pool                                             │
  ├─ enc2: ConvBlock(64  -> 128)  ──────────────┐      │
  │   pool                                      │      │
  ├─ enc3: ConvBlock(128 -> 256)  ───────┐      │      │
  │   pool                               │      │      │
  ├─ enc4: ConvBlock(256 -> 512)  ─┐     │      │      │
  │   pool                         │     │      │      │
  └─ bottleneck: ConvBlock(512 -> 1024)  │      │      │
      up + concat ─────────────────┘     │      │      │
      dec4: ConvBlock(1024+512 -> 512)   │      │      │
      up + concat ───────────────────────┘      │      │
      dec3: ConvBlock(512+256 -> 256)           │      │
      up + concat ──────────────────────────────┘      │
      dec2: ConvBlock(256+128 -> 128)                  │
      up + concat ─────────────────────────────────────┘
      dec1: ConvBlock(128+64 -> 64)
      输出 Conv1x1(64 -> 1) + Sigmoid
```

ConvBlock = Conv3x3(padding=1) + BN + ReLU，重复两次。`padding=1` 不改变边长，这样解码器和编码器的特征图尺寸才能对上。

| 配置 | 参数量 |
|---|---|
| base_channels=64（默认） | 31,036,481 |
| base_channels=32 | 7,762,465 |

### 跳跃连接解决的是位置信息的问题

编码器每下采样一次分辨率减半，到瓶颈层只剩原始尺寸的 1/16，语义强但这个任务需要逐像素修改，正常传播显然过于模糊。跳跃连接把编码器的高分辨率特征直接送到解码器，适合解决这个问题。

### 输出为什么接 Sigmoid

目标图是 [0,1] 的灰度，Sigmoid 把输出约束在同一区间，避免出现负值或超过 1 的像素。

---

## 损失函数

MSE / L1 / L2 或组合，用 `--loss` 切换：

| 名称 | 内容 | 说明 |
| --- | --- | --- |
| `L1` | 逐像素绝对值误差 | 默认 |
| `mse` | 逐像素平方误差（等价 L2） | 对边缘不友好 |
| `L1_grad` | L1 + 梯度（边缘）损失 | `--grad-weight` 控制权重 |

### L1 与 MSE

MSE 放大大的误差、对小误差宽容，优化时倾向把误差摊平到所有像素上，结果是边缘
变软、笔迹边界留下灰影。

---

## PSNR 与 SSIM 指标

| 指标 | 衡量 | 盲区 |
| --- | --- | --- |
| PSNR | 逐像素均方误差（对数化，单位 dB） | 对略微模糊不敏感 |
| SSIM | 亮度、对比度、结构三个角度的相似度 | 对逐像素的细小噪声不如 PSNR 敏感 |

两者互补恰好覆盖这个任务的核心风险：如果模型把印刷字也一起擦掉了，PSNR 会因为背景大面积一致而虚高，但 SSIM 会明显掉下来。所以两个一起报，互相制约。

---

## 全白 / 全黑

输出退化成常数是这类任务最典型的失败模式：网络发现「输出一片浅灰」能在 L1 损失下拿到不错的平均分，于是干脆不学任何结构。所以每个 epoch 的验证都记录预测图的亮度和标准差：

- 标准差低于 5e-3 判定为「近似纯色」，计数并计入 history 的 `val_flat`
- 训练时只要出现就立刻在终端警告
- `evaluate.py` 里把它作为一条验收标准明确标记

## 超参数

| 项目 | 值 |
| --- | --- |
| 输入 | 灰度单通道，原分辨率随机裁剪 384×384（训练） |
| base_channels | 64 |
| depth | 4 |
| 上采样 | 转置卷积（`--bilinear` 可换成双线性插值 + 卷积） |
| 损失 | L1（`--loss` 可切 mse / l1_grad） |
| batch size | 16 |
| 优化器 | AdamW，lr = 2e-4，weight_decay = 1e-5 |
| 学习率调度 | 余弦退火 |
| 梯度裁剪 | 1.0 |
| epochs | 60 |
| 数据增强 | 默认关闭，`--augment` 开随机缩放抖动 0.85~1.15 |
| 每轮验证张数 | 20 张整图（整图评估慢，训练中途抽样；最终评估跑全部） |
| 随机种子 | 42 |

U-Net 这种深层编解码结构里，转置卷积和 BatchNorm 组合下偶发的大梯度会让某一轮把权重带偏，需要梯度裁剪。

## 运行

环境准备见仓库根 README，需先激活环境。

```bash
# 完整训练（默认 L1 损失）
python level4_unet/src/train.py --data-root /path/to/dataset --epochs 60

# 云上被中断后续训（恢复模型、优化器状态和 epoch，并估算费用）
python level4_unet/src/train.py --data-root /path/to/dataset \
    --epochs 60 --tag unet_l1 --resume --gpu-hourly-cost 1.5

# 损失函数对照实验
python level4_unet/src/train.py --data-root /path/to/dataset --loss mse     --tag unet_mse
python level4_unet/src/train.py --data-root /path/to/dataset --loss l1_grad --tag unet_l1grad

# 显存不够时分块推理
python level4_unet/src/train.py --data-root /path/to/dataset --tile 512

# 在测试集上评估
python level4_unet/src/evaluate.py --checkpoint checkpoints/unet_l1_best.pt --worst 4

# 单张 / 批量推理，输出擦除后的干净图
python level4_unet/src/infer.py --checkpoint checkpoints/unet_l1_best.pt --input photo.jpg
python level4_unet/src/infer.py --checkpoint checkpoints/unet_l1_best.pt \
    --input-dir /path/to/dataset/20250213/dataset/input --limit 10
```

训练产出：

| 产物 | 路径 |
|---|---|
| 最优权重（验证 PSNR 最高） | `checkpoints/<tag>_best.pt` |
| 最近一轮 + 优化器状态（续训用） | `checkpoints/<tag>_last.pt` |
| Loss / PSNR / SSIM 演化曲线 | `reports/figures/<tag>_curves.png` |
| 输入/目标/预测/误差 四列对比图 | `reports/samples/unet_samples.png` |
| 全部超参、指标与逐轮 history | `reports/metrics/<tag>.json` |

评估与推理产出：

| 产物 | 路径 |
|---|---|
| 测试集指标（含逐图 PSNR/SSIM） | `reports/metrics/<tag>_test.json` |
| 最差 / 最好若干张的对比图 | `reports/figures/<tag>_test_{worst,best}.png` |
| 擦除后的干净图（PNG） | `reports/samples/<名称>_pred.png` |
| 单张的四列对比图（有真值时） | `reports/samples/<名称>_compare.png` |

---

## 验收标准对照

| 验收标准 | 对应实现 | 状态 |
|---|---|---|
| 输出 PSNR / SSIM 演化曲线 | `viz.plot_training_curves`，每轮验证都算 PSNR / SSIM 并记入 history | 已完成：`reports/figures/unet_l1_curves.png` |
| 结果不出现全白 / 全黑 | 每轮统计预测图标准差，`n_flat` 计数并警告；`evaluate.py` 里作为验收项打勾 | 已完成：验证 60 轮 0 次退化；test 630 张中 1 张触发阈值，但该张真值本身就是近空白页（详见下） |
| 记录训练时长与费用 | `train.py` 记录 `train_time_sec` / `train_time_hour`；`--gpu-hourly-cost` 自动算 `estimated_cost` | 时长已记录：39.0 分钟；费用按用户要求本次未估算 |

## 对比结果

数据来源：`reports/metrics/unet_l1.json` 与 `unet_l1_test.json`。

### 主实验

| 指标 | 值 |
|---|---|
| 模型 / 参数量 | U-Net base=64（31,036,481 参数） |
| 训练轮数 / 最优 epoch | 60 / 36 |
| 验证 PSNR（最优） | 22.00 dB |
| 验证 SSIM（最优） | 0.9409 |
| 测试集 PSNR | 23.74 dB（最差 8.66 / 最好 54.52） |
| 测试集 SSIM | 0.9584（最差 0.5225 / 最好 0.9984） |
| 全白/全黑检查 | 验证 60 轮退化 0 次；test 1/630 近白，该张真值 std 仅 0.0017、本身即空白页，预测 PSNR 54.52 dB 为全集最高，非退化失败 |
| 训练时长 | 0.65 小时（39.0 分钟） |
| 预估费用 | 本次未估算 |

### 损失函数对照

| 损失 | 测试 PSNR | 测试 SSIM | 备注 |
|---|---|---|---|
| L1 | 23.74 dB | 0.9584 | 默认，本次主实验 |
| MSE | 本次未做 | 本次未做 | |
| L1 + 梯度 | 本次未做 | 本次未做 | |

---

## 改进方向

题目要求思考 GAN / Diffusion 等方向，记录在这里供后续展开：

1. **对抗损失（pix2pix 式）**。目前的 L1/L2 都是逐像素度量，倾向于输出「平均意义上正确」的结果，笔迹残留和印刷字模糊在像素误差上代价相当。加一个判别器让模型去「骗过人眼」，能显著提升视觉锐度。代价是训练不稳定、需要调权重。

2. **感知损失**。用预训练网络（如 VGG）的特征距离替代部分像素损失，让模型优化人眼更在意的结构相似性，而不是逐像素吻合。

3. **可微 SSIM 损失**。目前 SSIM 只用于评估（skimage 的实现不可导）。换成可导版本直接当损失优化，能让训练目标和评估指标对齐。

4. **更强的数据增强**。当前的随机裁剪与缩放抖动比较保守，可以加弹性形变、局部亮度/对比度扰动，模拟扫描件的光照不均与纸张变形。

5. **多尺度 / 高分辨率策略**。先在 1/2 分辨率上训练收敛，再在原分辨率上微调 ——
   整图 2000 px 直接训练代价很高。

6. **Diffusion 类方法**。擦除任务可以当作条件生成来做，扩散模型在细节恢复上
   通常优于纯回归，代价是推理慢很多（多次迭代）。

---

## 本目录文件

| 文件 | 说明 |
|---|---|
| `README.md` | 本文档 |
| `configs/README.md` | 说明为什么用命令行参数而不是配置文件 |
| `src/data.py` | 配对数据集、尺寸处理、按日期划分 |
| `src/model.py` | U-Net（编码器-解码器 + 跳跃连接） |
| `src/losses.py` | L1 / MSE / L1+梯度 三种损失 |
| `src/metrics.py` | PSNR / SSIM 与「退化成纯色」检查 |
| `src/predict.py` | 尺寸补齐、分块推理、整集评估 |
| `src/viz.py` | 训练曲线与四列对比图 |
| `src/engine.py` | 训练循环（含断点续训与费用估算） |
| `src/train.py` | 训练入口（命令行接口） |
| `src/infer.py` | 单张 / 批量推理 |
| `src/evaluate.py` | 测试集评估与验收标准检查 |
