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
