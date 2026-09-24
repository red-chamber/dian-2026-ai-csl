# 学习记录

---

## 2026-09-15 ~ 09-16 ｜ Level 0：环境管理

### 目标

按题目要求，在 Windows 上通过 WSL2 使用 Linux，安装 Miniconda 做环境管理，建立一个独立、可复现、支持 GPU 加速的 PyTorch 环境，并用 `torch.cuda.is_available()` 验证它真的可用。

### 学到的概念

1. Conda 的作用

1). 不同项目的依赖不会冲突，隔离的环境类似于沙箱，不会对系统造成影响。
2). 可以让他人方便地复现项目的环境。使用 `conda env export` 导出 `environment.yml`，复现者使用 `conda env create` 构建即可。

2. 项目放在 Linux 文件系统里

项目路径是 `/home/lenovo/projects/dian-2026-ai-csl`，而不是挂载 `/mnt/c/...`。因为 WSL 访问 `/mnt/c` 大量小文件读写会明显变慢。

### 遇到的问题

1. Git 报 `Please tell me who you are`

**原因**：没配置提交身份。
**解决**：

```bash
git config --global user.name "用户名"
git config --global user.email "邮箱"
git config --global init.defaultBranch main
```

2. push 没有权限

**原因**：没有设置 SSH 密钥
**解决**：

```bash
ssh-keygen -t ed25519 -C "邮箱"
cat ~/.ssh/id_ed25519.pub        # 把公钥内容贴到 GitHub 的 SSH keys 里
ssh -T git@github.com            # 验证是否通了
```

### AI 使用记录

AI 起草了 `check_env.py`，以及 Level 0 的环境说明 `README.md`，以及解决一些报错

---

## 2026-09-18/2026-09-19 ｜ Level 1：model.py/train.py 的理解

### 目标

读懂两份代码，包括里面各个函数、模块的功能，了解其特殊写法与实现方式

### 学到的概念

1. 训练循环的步骤

1). `optimizer.zero_grad()` 清空上一轮累积的梯度
2). `logits = model(images)` 前向计算，得到 10 个类别的打分
3). `loss.backward()` 反向传播，用链式法则算出每个参数的梯度
4). `optimizer.step()` 按梯度更新参数

2. `@torch.no_grad()`

评估不需要反向传播，关掉计算图记录可以省显存、跑得更快

3. 只保存验证集上最好的权重

例如训练准确率还在涨，验证准确率反而下降，说明过拟合了

### 遇到的问题

1. 为什么 AI 写了命令行参数

想试不同学习率，如果直接改代码要在源码里改，容易污染 Git 历史

### 实测结果

| 项目 | 值 |
|---|---|
| 测试集准确率 | 98.16%（验收要求 ≥90%） |
| 最优 epoch | 第 7 轮，验证准确率 97.98% |
| 参数量 | 535,818 |
| 训练时长 | 约 33 s |
| GPU 显存峰值 | 28.9 MB |

### AI 使用记录

`model.py`、`train.py`、`infer.py` 都由 AI 起草。我阅读时逐行加了注释，并删掉用不到的调试参数。

---

## 2026-09-20 ｜ Level 1：infer.py 的理解

### 目标

读懂推理脚本，重点是搞清楚"推理"和"训练"到底差在哪，以及一个 `.pt` 文件怎么变回可用的模型。

### 学到的概念

1. While inferring:

1). `model.eval()` 关掉 Dropout，固定行为
2). `@torch.no_grad()` 关掉自动求导，不建计算图，省显存。

2. What are saved in checkpoint(ckpt)?

| 类别 | 键 | 作用 |
|---|---|---|
| 权重 | `model_state` | 6 个张量，535,818 个参数，占文件体积 99% |
| 结构 | `model_config` | 重建模型 |
| 元信息 | `epoch` / `val_acc` / `test_acc` / `args` | 确认实验版本 |

3. Four steps in loading weights:

1). `torch.load(path, map_location=device)` —— 让 GPU 上存的权重也能在 CPU 机器上读出来，标准化写法
2). `build_model_from_config(ckpt["model_config"])` —— 加载整体结构
3). `model.load_state_dict(ckpt["model_state"])` —— load weights
4). `model.to(device)` and `model.eval()` —— 导入设备，开启评估模式

4. Why the weight files are named ".pt"?

It's a standard naming convention standing for PyTorch.Likely,".pth" is also right.

### 遇到的问题

1. `--index 0` cannot be written as `if args.index:`

`0` 在 Python 里是 falsy，第一次 index 为 0 时必然发生错误。

解决：`if args.index is not None:` 是正确且规范的写法

### 实测结果

| 项目 | 值 |
|---|---|
| 推理命令 | `python level1_mlp/src/infer.py --index 0` |
| 输出 | 终端打印 10 个类别的概率与条形图 |
| 结果图 | `reports/samples/mlp_infer_test_00000.png`（左原图 / 右概率条形图） |

### AI 使用记录

`infer.py` 由 AI 起草。我逐段阅读时向 AI 追问了每一处的语法与设计理由，并加上了白底图片的反色逻辑。

---

## 2026-09-20 ｜ Level 2：CNN model.py 的理解

### 目标

读懂 CNN 的定义

### 学到的概念

1. The biggest difference from MLP: keeping the 4D shape

MLP 拿到 `(N, 1, 28, 28)` 的第一件事是 `Flatten`，立刻压成 `(N, 784)`，退化成一维，但CNN 全程保持 `(N, C, H, W)` 四维，直到分类前才展平

2. Where the shape `(N, 1, 28, 28)` comes from

| 维度 | 值 | 来源 |
|---|---|---|
| N | 128 | `DataLoader(batch_size=128)` 一次取多少张 |
| 1 | 1 | MNIST 只有一个通道 |
| 28, 28 | 28×28 | 数据集本身的图片尺寸 |

3. Kaiming 初始化的原理：让每层输出的方差保持不变

`y = Wx + b`，假设 w、x 零均值且独立，则 `Var(y) = fan_in · Var(w) · Var(x)`。
要 `Var(y) = Var(x)`，需要 `std(w) = 1/sqrt(fan_in)`。
但 ReLU 把负半轴置零，方差减半，所以要再补一个 2 倍

### 遇到的问题

1. 卷积核的初始权重比全连接层大，以为写错了

`Conv2d(1, 32, 3)` 初始化出来的权重 std 约 0.47，而 `Linear(3136, 32)` 只有 0.025，差了近 20 倍。
原因：`fan_in` 越小，为了维持方差稳定，单个权重就需要越大，所以这是正常的

### 实测结果

直接运行 `python level2_cnn/src/model.py`，核对结构与形状流：

| 项目 | 值 |
|---|---|
| 可训练参数量 | **119,530**（与 docstring 手算式一致） |
| 输入形状 | `(4, 1, 28, 28)` |
| 卷积 + 池化后 | `(4, 64, 7, 7)` |
| 展平后 | `(4, 3136)`（= 64×7×7） |
| 输出形状 | `(4, 10)`，每张图 10 个类别的打分 |

### AI 使用记录

CNN 的 `model.py` 由 AI 起草。我阅读时逐段追问了形状变化、`padding` 的作用和 Kaiming 初始化的推导。

---

## 2026-09-20 ｜ Level 2：train.py / infer.py / compare.py 的理解

### 目标

读懂剩下三个脚本

### 学到的概念

1. train.py 与 Level 1 只差三处

1). `MLP` 改成 `from model import CNN`
2). 命令行参数 `--conv-channels` / `--fc-hidden` 改成 `--hidden-sizes`
3). CNN build model 时显式传 `in_channels=1, input_size=28`

因为前面定义过了 `nn.Module` 的通用接口，这里全部调用这个，方便且风格统一。

| 接口 | 用途 |
| --- | --- |
| `model(images)` | 前向计算 |
| `model.parameters()` | 交给优化器 |
| `model.state_dict()` | 存权重 |
| `model.train()` / `model.eval()` | 切换模式 |

2. `shuffle=False` 之必要性

`collect_predictions` 返回的数组，下标必须和数据集下标严格一一对应，因为后面要用下标取原始图片，所以测试时不能用 `shuffle=True` 打乱顺序，训练时可以用这个

3. 混淆矩阵

`cm[i][j]` 意为真实是 i、却被预测成 j 的样本数，其中 i，j 都表示类别
- **对角线**意为猜对了（`i == j`）；**非对角线**意为猜错了，把 ij 两类混淆了。
- **错误数** = 全部元素和 − 对角线之和
- 准确率的局限性：准确率是一个被压扁的标量，没有保留被混淆的具体的类别的信息

10. 置信度的含义及作用

置信度意为模型对结果的“自信程度”。错分样本中，置信度低说明它在犹豫，置信度高说明错得离谱

### 遇到的问题

1. 错误样本图显示成噪声

**原因**：用了带归一化 transform 的数据集来显示图片，像素值有正有负
**解决**：额外加载一份不带 transform 的 `raw_test` 专门用于显示

### 实测结果

| 项目 | 结果 |
| --- | --- |
| `model.py` | `(4,1,28,28)` → `(4,64,7,7)` → `(4,3136)` → `(4,10)` |
| `compare.py` | `confusion_matrix`（错误数 = 总和 − 迹）、`top_confusions`、`epochs_to_reach` |
| 三个脚本端到端 | `checkpoints/cnn_mnist_best.pt`、`reports/metrics/cnn_mnist.json` |

### AI 使用记录

`train.py`、`infer.py`、`compare.py` 都由 AI 起草。我阅读时重点追问了混淆矩阵的计算方式和错误样本的分组逻辑

---

## 2026-09-20 ｜ matplotlib 画图

### 1. 每个绘图函数的流程

```python
out_path.parent.mkdir(parents=True, exist_ok=True)   # 1. 确保目录存在
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))    # 2. 建立画布
# Process                                            # 3. 画
fig.suptitle(title, fontsize=13)                     # 4. 总标题
fig.tight_layout()                                   # 5. 收紧布局，防止重叠
fig.savefig(out_path, dpi=150)                       # 6. save，关闭并释放内存
plt.close(fig)
```

### 2. `matplotlib.use("Agg")`

```python
matplotlib.use("Agg")          # Agg 意为只写文件，不弹窗，TkAgg 等是弹窗口
```

### 3. `savefig`

```python
# 保存时加这行是因为目录不存在时需要建立
out_path.parent.mkdir(parents=True, exist_ok=True)
```

### 4. 用 `fig.savefig()` 而非 `plt.savefig()`

`plt.savefig()` 存的是当前活跃的 figure，`fig.savefig()` 明确指定存哪张

### 5. `plt.close(fig)`

matplotlib 会一直持有 figure 对象。如果每个 epoch 存一张，不关闭会持续累积内存，超过 20 张时有 `RuntimeWarning: More than 20 figures have been opened`。所以 `plt.close(fig)` 用完一张就释放

### 6. matplotlib 接收的必须是 numpy / Python 原生类型

```python
axes[1].bar(range(10), probs.numpy())        # 张量 -> numpy
for i, p in enumerate(probs.tolist()):       # 张量 -> Python float 列表
```

### 7. 图上标注数据

```python
axes[1].text(i, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
ax.text(j, i, str(cm[i, j]), ha="center", va="center", ...)
```

`ha` / `va` 是水平/垂直对齐，`ha="center"` 让文字以坐标点为中心，`p + 0.02` 加偏移量，避免文字与柱顶重合

### 14. `figsize` 和 `dpi` 共同决定输出像素

```python
plt.subplots(1, 2, figsize=(12, 4.5))
fig.savefig(out_path, dpi=150)   
```

`figsize` 单位是**英寸**，是「设想中的物理尺寸」，`dpi` 是每英寸多少像素，`12 × 150 = 1800`，两者相乘才是最终像素数。

---

## 2026-09-23 ｜ Level 3：四个文件的理解

### 目标

读懂 `level3_classic_networks/src/` 下的四个文件

### 学到的概念

1. AlexNet 的改造：保留结构特征，只改尺寸

原版面向 224×224 的 ImageNet，直接搬过来在 28×28 上用不了。改造了三处 :

| 原版 | 改造后 |
| --- | --- | 
| 第一层 11×11 卷积、步长 4 | 5×5、步长 1、padding=2 |
| 全连接 4096 → 4096 | 1024 → 512 |
| 池化在第 2、5 层之后 | `pool_after = (0, 1, 4)` |

2. AlexNet 的参数量构成：主要在分类头

`Linear(2304, 1024)` 参数量 2,360,320，占整网的 44.2%。原因和 Level 2 的 CNN 一样：全连接层的参数量是「输入维 × 输出维」，展平后 2304 维的输入会让它迅速膨胀。

3. ResNet 的三处设计

stem 不做下采样。原版是 7×7 卷积加步长 2 池化，靠它把 224 一次降到 56。28 像素直接复用不现实，所以改成 CIFAR 版写法，一个 3×3 卷积

BasicBlock 的跨层连接。结构: Conv3x3 → BN → ReLU → Conv3x3 → BN → 加 → ReLU

全局平均池化。`AdaptiveAvgPool2d(1)` 把 `(N, 512, 4, 4)` 压成 `(N, 512, 1, 1)`

4. 跨层连接的两种形式

代码里按形状自动选择：

```python
self.shortcut = nn.Identity()          # 形状能直接相加时用恒等映射
if stride != 1 or in_channels != out_channels:
    self.shortcut = nn.Sequential(     # 否则用1×1卷积+BN调整形状
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    )
```

通道数变了（64 → 128），或者 `stride=2` 让特征图变小了，此时需要使用 `1*1` 卷积 modify 形状，使其能正确输入到下一步。
为什么是 `1*1` 卷积？因为只有它可以实现只变换通道数，而不变动空间形状，它同时也下采样（`stride=2`）。下采样是跨层连接里唯一丢失少量空间特征的地方。

5. 残差连接不引入任何参数

`residual=True` 和 `residual=False` 的参数量完全相同，都是 11,172,810。原因是 `out + identity` 是逐元素加法，加法本身没有可学习参数，正如上面一点说的，只有下采样会丢失少量细节。

因此赋 redidual=False 只有一个变量，适合作消融实验。

### 遇到的问题

1. 残差消融的结果与预期相反

`residual=True` 的 ResNet-18 测试准确率 92.63%，去掉跨层连接的 PlainNet-18 是 92.76%，两者大概相同，而且测试损失明显更低（0.2273 vs 0.3767），到达最优更早。

按题目和论文的预期，残差连接应该带来提升，但是结果相反，三点推测：

1) 18 层还不够深。原论文中网络退化出现在 34 层以上，这里只有 16 个卷积层，且输入仅 28×28，梯度还不足以衰减到需要「直通道」的程度。
2) BatchNorm 已经承担了稳定梯度尺度的主要工作。
3) 两个模型都严重过拟合，比较被正则化效应主导。

2. 两个模型都严重过拟合

ResNet-18 第 20 轮训练准确率 99.46%，最优验证准确率 93.30%，相差 6.16 个百分点。1117 万参数对 54,000 张训练图来说过多，这个任务使用 ResNet18 仍然过于简单。

### 实测结果

| 指标 | AlexNet | ResNet-18 | PlainNet-18 |
| --- | --- | --- | --- |
| 参数量 | 5,338,314 | 11,172,810 | 11,172,810 |
| 测试准确率 | 91.42% | 92.63% | 92.76% |
| 测试损失 | 0.2706 | 0.3767 | 0.2273 |
| 第 1 轮验证准确率 | 86.53% | 89.12% | 87.03% |
| 达到 92% 验证准确率 | 未达到 | 第 4 轮 | 第 6 轮 |
| 最优 epoch / 验证准确率 | 第 14 轮 / 91.98% | 第 19 轮 / 93.30% | 第 9 轮 / 92.95% |
| 第 20 轮训练准确率 | 95.60% | 99.46% | 99.15% |
| 每轮平均耗时 | 7.2 s | 22.2 s | 22.0 s |
| GPU 显存峰值 | 257.6 MB | 659.0 MB | 637.2 MB |

三个数字值得记住：

- ResNet-18 用 2.09 倍参数量只换来 1.21 个百分点的提升，而每轮耗时是 AlexNet 的 3.1 倍。
- 残差消融是负结果（92.63% vs 92.76%），原因尚未查清。
- 参数量最大的 ResNet-18 过拟合最严重，验证准确率与训练准确率差 6.16 个百分点。

### AI 使用记录

Level 3 的 `model.py`、`train.py`、`infer.py`、`compare.py` 以及 `common/` 包都由 AI 起草。我逐个文件读完后向 AI 追问了 `sys.path` 引导的必要性、1×1 卷积在跨层连接里的作用、全局平均池化为什么能消掉分类头参数，以及残差消融为什么没出现预期结果。

## 2026-09-24 ｜ Level 4：U-Net 手写内容擦除

### 目标

读懂 `level4_unet/src/` 下的十个文件（主要是其中的三个），理解 U-Net 的编码器-解码器与跳跃连接，完成训练、评估与真实图片推理

### 学到的概念

1. 为什么用原分辨率裁剪而不是整图缩放

这个任务要求输出与输入逐像素对齐，擦除结果要能贴回原图，缩放会破坏这个对应关系。并且原分辨率随机裁剪顺带相当于数据增强，部分弥补了数据量小的问题。

2. 数据集质量不齐

| 情况 | 处理方式 |
| --- | --- |
| 扩展名是 .jpg 但内容是带 alpha 的 PNG | 先合成到白底，再转灰度 |
| 长边从 143 px 到 2000 px，长宽比各异 | 训练用原分辨率随机裁剪；验证测试用整图 |
| 尺寸不是 16 的整数倍 | 推理前补齐，跑完裁回原尺寸 |

3. 输入边长为什么必须是 2^depth 的整数倍

每次下采样 `MaxPool2d(2)` 遇到奇数边长会向下取整，丢掉最边上一两个像素，上采样之后就和跳跃连接的尺寸差 1~2 像素，会报错，所以需要使用 `predict.py` 在推理前把图补齐到 16 的整数倍。

4. U-Net 的形状守恒

| 阶段 | 形状 |
| --- | --- |
| 输入 | (N, 1, 256, 256) |
| enc1 → pool | (N, 64, 256, 256) → (N, 64, 128, 128) |
| enc2 → pool | (N, 128, 128, 128) → (N, 128, 64, 64) |
| enc3 → pool | (N, 256, 64, 64) → (N, 256, 32, 32) |
| enc4 → pool | (N, 512, 32, 32) → (N, 512, 16, 16) |
| bottleneck | (N, 1024, 16, 16) |
| up1 → cat → dec4 | (N, 512, 32, 32) → (N, 1024, 32, 32) → (N, 512, 32, 32) |
| up2 → cat → dec3 | (N, 256, 64, 64) → (N, 512, 64, 64) → (N, 256, 64, 64) |
| up3 → cat → dec2 | (N, 128, 128, 128) → (N, 256, 128, 128) → (N, 128, 128, 128) |
| up4 → cat → dec1 | (N, 64, 256, 256) → (N, 128, 256, 256) → (N, 64, 256, 256) |
| head + Sigmoid | (N, 1, 256, 256) |

所以输出形状和输入完全相同。4 次池化减半和 4 次转置卷积翻倍正好抵消。

5. 跳跃连接传递的是位置信息

编码器每下采样一次分辨率减半，到瓶颈层只剩 1/16，语义强但像素位置关系已经模糊，需要 skip connection 传递位置信息。

6. PSNR 和 SSIM 必须一起看

| 指标 | 衡量 | 盲区 |
| --- | --- | --- |
| PSNR | 逐像素均方误差，并对数化成 dB | 对略微模糊不敏感 |
| SSIM | 亮度、对比度、结构三个角度的相似度 | 对逐像素的细小噪声不敏感 |

如果把印刷字也擦掉，PSNR 会因为背景大面积一致而变高，但 SSIM 会明显降低，两者需要互相制约

7. 退化检测

网络发现输出一片浅灰的分数比较高，容易使学到的结构退化或消失。所以每个 epoch 都记录预测图的标准差，低于 5e-3 判定为近似纯色并计数。

8. 三种损失

| 损失 | 优化到的统计量 | PSNR | SSIM |
| --- | --- | --- | --- |
| MSE | 条件均值 | 最高 | 最低 |
| L1 | 条件中位数 | 中 | 中 |
| L1 + 梯度 | 中位数 + 边缘对齐约束 | 最低 | 最高 |

MSE 在 PSNR 上占优，因为PSNR 就是 MSE 的对数变换，用 MSE 当损失等于直接优化评估指标本身。它的问题在于解是条件均值，但这个任务的条件分布是双峰的，要么是字要么是背景，使用它容易输出灰色。L1 的解是条件中位数，会倒向样本更多的那一侧，输出更干净但偏离均值。

### 遇到的问题

1. 全图直推的做法不合适，需要分块

本机 WSL 只有 7.6 GB 内存，全图直推时尺寸过大，进程被杀。改用分块后峰值降到 200 MB 以内。

2. L1 PSNR 从第 20 轮起就几乎停止

L1 的 PSNR 在 21.7~22.0 dB 之间震荡了 40 轮没有提升，损失虽然在下降但是减少得很少。MSE 和 L1+梯度的最优轮次都在第 44 轮，L1 在第 36 轮，说明 L1 收敛得早，数据量太少是主要瓶颈，而非参数量。

### 实测结果

主实验（L1 损失）

| 项目 | 值 |
| --- | --- |
| 模型 / 参数量 | U-Net base_channels=64 depth=4 / 31,036,481 |
| 数据划分 | train 1712 对 / val 70 对 / test 630 对，按日期留出 |
| 最优 epoch | 第 36 轮（验证 PSNR 22.00 dB / SSIM 0.9409） |
| 测试 PSNR | 23.74 dB（最差 8.66 / 最好 54.52） |
| 测试 SSIM | 0.9584（最差 0.5225 / 最好 0.9984） |
| 退化检查 | 验证 60 轮 0 次；测试 1/630 触发阈值 |
| 训练时长 / 显存峰值 | 39.0 分钟（2342.5 s）/ 13,300.8 MB |

损失函数对照（同配置，只有 `--loss` 不同）

| 损失 | 最优 epoch | 验证 PSNR / SSIM | 测试 PSNR | 测试 SSIM | 训练时长 |
| --- | --- | --- | --- | --- | --- |
| L1 | 36 | 22.00 / 0.9409 | 23.74 dB | 0.9584 | 39.1 分钟 |
| MSE | 44 | 23.12 / 0.9255 | 24.86 dB | 0.9493 | 39.1 分钟 |
| L1 + 梯度 | 44 | 22.18 / 0.9508 | 24.31 dB | 0.9620 | 39.2 分钟 |

自定义图片推理（本机 CPU，分块推理块边长 512）

| 图片 | 尺寸 | 预测标准差 | 改动像素比例 | 耗时 |
| --- | --- | --- | --- | --- |
| PixPin_2026-09-24_13-30-46 | 1997 × 2455 | 0.1600 | 6.48% | 约 1 分钟 |
| PixPin_2026-09-24_13-31-31 | 1974 × 2620 | 0.1551 | 6.74% | 约 1 分钟 |

### AI 使用记录

Level 4 的十个源文件由 AI 起草，训练在远端 GPU 上完成，训练与评估脚本、文档回填、以及本机的推理由 AI 执行。我逐个文件读完后追问并尝试理解。
