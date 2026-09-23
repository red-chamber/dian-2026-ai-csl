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

读懂 `level3_classic_networks/src/` 下的四个文件，重点是两个经典网络的结构设计、
以及它们各自「把参数花在哪」的差别。顺带读懂新出现的 `common/` 包是怎么把三个
Level 的训练流程合并成一份的。

### 学到的概念

1. `common/` 是怎么接上的：脚本要自己把仓库根塞进 sys.path

按 `python level3_classic_networks/src/train.py` 这种方式运行时，Python 只把
脚本自己所在的目录放进 `sys.path`，仓库根不在里面。所以 `import common.data` 会失败。
每个入口脚本开头都补了三行：

```python
_ROOT = Path(__file__).resolve().parents[2]   # 往上两级到仓库根
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
```

`parents[2]` 是因为文件在 `<仓库根>/levelN/src/xxx.py`：`parents[0]` 是 `src`，
`[1]` 是 `levelN`，`[2]` 才是仓库根。`model.py` 里也有一份，因为它要 import
`common.utils` 拿 `count_parameters`。

好处是三个 Level 的 `train.py` 现在共用 `common/engine.py` 里的 `run_training`，
训练循环、保存最优权重、测试集评估、补写 `test_acc` 这套流程只维护一份。

2. AlexNet 的改造：保留结构特征，只改尺寸

原版面向 224×224 的 ImageNet，直接搬过来在 28×28 上用不了。改造了三处，
每一处都是因为「图像变小了」：

| 原版 | 改造后 | 原因 |
|---|---|---|
| 第一层 11×11 卷积、步长 4 | 5×5、步长 1、padding=2 | 11×11 加步长 4 在 28 像素上一次就吃掉半张图 |
| 全连接 4096 → 4096 | 1024 → 512 | 缩小规模，但仍是整网的参数大头 |
| 池化在第 2、5 层之后 | `pool_after = (0, 1, 4)` | 得到 28 → 14 → 7 → 3 的节奏 |

`pool_after` 这个写法的好处是池化位置和展平维度的推算共用同一个集合，
不会出现「改了池化位置但忘了改 `flat_dim`」的不一致。

3. AlexNet 的参数量构成：大头在分类头

```
卷积部分（5 层）      2,448,064   45.9%
全连接部分（2 层）    2,885,120   54.0%
输出层                    5,130    0.1%
合计                  5,338,314
```

其中 `Linear(2304, 1024)` 一层就是 2,360,320，占整网的 44.2%。
原因和 Level 2 的 CNN 一样：全连接层的参数量是「输入维 × 输出维」，
展平后 2304 维的输入会让它迅速膨胀。

这是 AlexNet 这类早期网络的典型特征 —— 卷积层不深，参数全堆在分类头。
后面 ResNet 用全局平均池化把这个大头直接削掉了。

4. ResNet 的三处设计

stem 不做下采样。原版是 7×7 卷积加步长 2 池化，靠它把 224 一次降到 56。
28 像素经不起这一刀，所以改成 CIFAR 版写法：一个 3×3 卷积，保持 28×28。

BasicBlock 的跨层连接。结构是「Conv3x3 → BN → ReLU → Conv3x3 → BN → 加 → ReLU」，
跨层连接把输入直接加到第二个 BN 的输出上。

全局平均池化。`AdaptiveAvgPool2d(1)` 把 `(N, 512, 4, 4)` 压成 `(N, 512, 1, 1)`，
所以不管前面特征图多大，送进全连接的都只有 512 维。

5. 两个网络的参数量分布正好相反

```
AlexNet     分类头 54.0%，卷积层 45.9%
ResNet-18   分类头  0.0%（只有 5,130，占 0.0005%），卷积层 99.9%
```

放在一起看，这是 Level 3 最值得记住的一个对比：ResNet 参数量更大（1117 万 vs 533 万），
但全部花在特征提取上；AlexNet 有一半参数在最后两层全连接里。
「参数从分类头搬到卷积层」这句话，在这两个数字上是直接看得见的。

6. 跨层连接的两种形式，以及 1×1 卷积为什么出现在这里

代码里按形状自动选择：

```python
self.shortcut = nn.Identity()          # 形状能直接相加时用恒等映射
if stride != 1 or in_channels != out_channels:
    self.shortcut = nn.Sequential(     # 否则用 1×1 卷积 + BN 调形状
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    )
```

两个触发条件：通道数变了（64 → 128），或者 `stride=2` 让特征图变小了。
两种情况下主路径输出的形状和输入对不上，加法没法做，所以要调整。

1×1 卷积在这里是唯一合适的选择，因为它是唯一能「只改通道数、不动空间尺寸」的操作。
它顺便还承担了下采样（`stride=2`）。这也是跨层连接唯一的额外参数开销。

7. 残差连接不引入任何参数

`residual=True` 和 `residual=False` 的参数量完全相同，都是 11,172,810。
原因是 `out + identity` 是逐元素加法，加法本身没有可学习参数。

这一点让残差消融成为一个干净的对照实验：两个模型的参数量、超参数、数据划分、
随机种子全部相同，唯一变量就是「有没有那条跨层通道」。

消融的实现方式也很轻：

```python
identity = self.shortcut(x) if self.residual else 0
```

一个 `bool` 参数从 `ResNet18` 透传到每个 `BasicBlock`，不需要改结构代码。
`train.py` 的 `--no-residual` 直接接上这个参数。

8. `config()` 里为什么要多一个 `arch` 字段

Level 1/2 只有一个模型，checkpoint 里的 `model_config` 不需要说明是哪个模型。
Level 3 有 AlexNet 和 ResNet 两个，所以 `config()` 里加了 `"arch": "alexnet"` 或
`"arch": "resnet"`，`build_model_from_config` 靠它分发：

```python
arch = config.get("arch", "alexnet")
if arch not in ARCHITECTURES: ...
```

这样 `infer.py` 不需要额外的 `--model` 参数 —— 把 `--checkpoint` 指到哪个权重，
就自动跑哪个模型。结构信息跟着权重文件走，这个思路从 Level 1 一路沿用下来。

9. `compare.py` 不重新训练

它只读各模型训练时导出的 `reports/metrics/<tag>.json`，所以跑一次只要几秒：

```python
result = load_metrics(tag)          # 直接读 json
history = result["history"]         # 逐轮曲线直接从 json 里拿
```

能用这种方式的前提是训练时把逐轮 `history` 存进了 json。
如果没存，对比脚本就得把每个模型重新训练一遍才能画收敛曲线。

收敛速度那项用 `epochs_to_reach(history, threshold)`，在 `history["val_acc"]` 里
找第一次越过阈值的轮次，找不到就返回 `None`（打印成「未达到」）。

另外 `load_all` 是用「缺哪个就跳过哪个并提示」的写法，而不是遇到缺失就整体失败 ——
只想对比两个模型时，不必先把第三个也训练出来。

10. Fashion-MNIST 的标签不是数字

MNIST 的 10 个类是数字 0~9，Fashion-MNIST 是 10 类衣物（T-shirt/top、Trouser、
Pullover……）。所以 `infer.py` 从数据集注册表里取 `spec["classes"]` 来显示类别名，
而不是直接打印 `0~9`。

类别名有长有短，画成竖排柱状图会把 x 轴标签挤在一起，所以改成了横向条形图
（`barh` + `set_yticklabels` + `invert_yaxis`，让第 0 类显示在最上面）。

11. 为什么 Level 3 换数据集

尺寸和通道数与 MNIST 完全一样（28×28 灰度），所以数据管道直接复用，
唯一要换的是归一化的均值和标准差（在 `common/data.py` 的注册表里）。

换的原因不是尺寸，是难度：MNIST 上 MLP 就能到 98%，所有模型都挤在 99% 以上，
网络结构的差异被抹平。Fashion-MNIST 的类间差异更细（T-shirt/Pullover/Coat/Shirt
这四类经常互相混淆），深层网络的价值才看得出来。
实测的 91%~93% 与 MNIST 的 98%~99% 对比，说明这个选择起作用了。

### 遇到的问题

1. 残差消融的结果与预期相反（这一 Level 最重要的发现）

`residual=True` 的 ResNet-18 测试准确率 92.63%，去掉跨层连接的 PlainNet-18 是 92.76%
—— 不降反升，而且测试损失明显更低（0.2273 vs 0.3767），到达最优更早（第 9 轮 vs 第 19 轮）。

按题目和论文的预期，残差连接应该带来提升，这里没有复现出来。给出的四点推测
（都未经进一步实验验证）：

- 18 层还不够深。原论文中网络退化出现在 34 层以上，这里只有 16 个卷积层，
  且输入仅 28×28，梯度还不足以衰减到需要「直通道」的程度。
- BatchNorm 已经承担了稳定梯度尺度的主要工作。这里的 PlainNet 同样含 BN，
  残差连接在 2015 年解决的部分问题如今很大程度上被 BN 覆盖了。
- 两个模型都严重过拟合，比较被正则化效应主导 —— 测到的差异更多反映
  「谁更晚过拟合」，而不是「谁的优化更充分」。
- 只有 seed 42 的单次运行，0.13 个百分点的差距不足以声称有统计显著性。

结论：这是个负结果，需要多组随机种子取均值、把深度加到 34/50 层、或换 CIFAR-10
才能变成有说服力的结论。这次的做法是把它如实写进实验记录并明确标注为推测，
而不是挑一个符合预期的说法。

2. 两个模型都严重过拟合

ResNet-18 第 20 轮训练准确率 99.46%，最优验证准确率 93.30%，相差 6.16 个百分点；
验证损失从第 4 轮的 0.1997 一路回升到第 20 轮的 0.3741，而训练损失同期从 0.1504
降到 0.0166。1117 万参数对 54,000 张训练图来说过多，训练集基本被记住了。

训练脚本按验证准确率保存最优权重，所以最终用的是第 19 轮的模型，
但即便如此，93.30% 说明单纯加深度在当前配置下已经收益有限。

3. AlexNet 差 0.02 个百分点没达到 92%

`compare.py` 的 `--target` 默认 0.92，判据是验证集准确率。
AlexNet 全程 20 轮最高只到 91.98%，所以在「达到 92% 所需 epoch」那行记为「未达到」。
这个数字容易误读成 AlexNet 很差，实际只差 0.02 个百分点，
文档里专门加了一句说明，避免被当成「完全没学到」。

### 实测结果

Fashion-MNIST，20 轮，seed 42，无数据增强（2026-09-22，commit `3862cd6`）：

| 指标 | AlexNet | ResNet-18 | PlainNet-18 |
|---|---|---|---|
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

Level 3 的 `model.py`、`train.py`、`infer.py`、`compare.py` 以及 `common/` 包都由 AI 起草。
我逐个文件读完后向 AI 追问了 `sys.path` 引导的必要性、1×1 卷积在跨层连接里的作用、
全局平均池化为什么能消掉分类头参数，以及残差消融为什么没出现预期结果。
代码注释我做了统一和精简，输出信息也改成了英文与中文混排的当前版本。
