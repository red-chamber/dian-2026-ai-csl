# Level 2：CNN 完成 MNIST 并与 MLP 对比

目标：把 MLP 换成 CNN，通过**对照实验**理解卷积为什么更适合图像任务

验收标准：CNN 测试集准确率约 96%，从**准确率 / 参数量 / 收敛速度 / 错误样本**四个角度与 MLP 对比

---

## 与 Level 1 的关系

代码结构完全复用 Level 1，**唯一的变化是模型定义**，MLP 是展平->全连接，CNN 是卷积->池化->全连接
超参数刻意保持一致，这样两组实验的差异**只来自网络结构本身**，对比才有意义。

---

## 网络结构

```text
  input                               (N, 1, 28, 28)
  Conv2d(1, 32, 3, padding=1) + ReLU  (N, 32, 28, 28)
  MaxPool2d(2)                        (N, 32, 14, 14)
  Conv2d(32, 64, 3, padding=1) + ReLU (N, 64, 14, 14)
  MaxPool2d(2)                        (N, 64, 7, 7)
  Flatten                             (N, 3136)
  Dropout
  Linear(3136, 32) + ReLU             (N, 32)
  Dropout
  Linear(32, 10)                      (N, 10)   logits
```

## 参数量主要在展平后的全连接层而非卷积层

数据表现：
- 两层卷积合计 18,816 个参数，而 `Linear(3136, 32)` 占 100,384，整体的 84%
- 所以 AI 的写法中第一个 fc 层只取 32 维，因为若取 128 维，这一层的参数量会变得很大，和 MLP 几乎持平，就看不出来 CNN 能用更少参数达到更高准确率这个特点了。

### Why `padding=1`

`kernel_size=3`，`padding=1` 时边长只由池化减半，padding 用来维持卷积的边长不变，达成 same padding

```text
边长变化：
28 --conv--> 28 --pool--> 14 --conv--> 14 --pool--> 7
```

### 为什么输出层同样不加 softmax

`nn.CrossEntropyLoss` 这个二维交叉熵损失内部其实已经有了 softmax

---

## 超参数

| 项目 | val |
| --- | --- |
| 数据集大小 | MNIST（train 54,000 / val 6,000 / test 10,000） |
| 输入尺寸 | 1 × 28 × 28 |
| 卷积通道 | 32 → 64 |
| fc_hidden | 32 |
| batch size | 128 |
| 优化器 | Adam，lr = 1e-3，weight_decay = 0 |
| 损失函数 | CrossEntropyLoss |
| epochs | 10 |
| dropout | 0.2 |
| 权重初始化 | Kaiming Normalization，偏置置零 |
| 随机种子 | 42 |
---

## 运行

环境准备见仓库根 README，需先 `conda activate dian-ai`。

```bash
# 1. 训练 CNN
python level2_cnn/src/train.py

# 2. 单张图片推理
python level2_cnn/src/infer.py --index 0

# 3. 四角度对比，前提是两个模型都已经训练
python level2_cnn/src/compare.py
```

产物：

| 产物 | 路径 |
|---|---|
| 最优权重 | `checkpoints/cnn_mnist_best.pt` |
| Loss / Accuracy 曲线 | `reports/figures/cnn_mnist_curves.png` |
| 完整指标与逐轮 history | `reports/metrics/cnn_mnist.json` |

对比产出：

| 产物 | 路径 |
|---|---|
| 收敛速度对比曲线 | `reports/figures/mlp_vs_cnn_curves.png` |
| 混淆矩阵并排 | `reports/figures/mlp_vs_cnn_confusion.png` |
| 错误样本分类网格 | `reports/samples/mlp_vs_cnn_errors.png` |
| 全部对比数字 | `reports/metrics/compare_mlp_cnn.json` |

---

## 四角度对比

> ⚠️ **以下数值待填**：先跑完 `train.py` 再跑 `compare.py`，把终端输出的数字和
> `reports/metrics/compare_mlp_cnn.json` 里的值抄进来。表格结构已经定好，不预填推测值。

### 角度 1｜参数量

| 模型 | 参数量 |
|---|---|
| MLP [512, 256] | 535,818 |
| CNN [32, 64] fc=32 | 119,530 |
| 比值 | CNN 只有 MLP 的 22.3% |

### 角度 2｜准确率

| 模型 | 测试集准确率 | 错误张数 |
|---|---|---|
| MLP | _待填_ | _待填_ |
| CNN | _待填_ | _待填_ |
| 提升 | _待填_ | |

### 角度 3｜收敛速度

| 指标 | MLP | CNN |
|---|---|---|
| 第 1 轮验证准确率 | _待填_ | _待填_ |
| 达到 96% 所需 epoch | _待填_ | _待填_ |
| 每轮平均耗时 | _待填_ | _待填_ |
| 最优 epoch / 验证准确率 | _待填_ | _待填_ |

### 角度 4｜错误样本

`compare.py` 会把测试集样本分成三类：

| 类别 | 张数 | 说明 |
|---|---|---|
| 两者都错 | _待填_ | 两个模型都搞不定的硬样本 |
| 只有 MLP 错 | _待填_ | **CNN 修好了的样本，是卷积带来收益的直接证据** |
| 只有 CNN 错 | _待填_ | MLP 对而 CNN 错的样本，用来审视 CNN 的短板 |

另外统计各自最容易混淆的数字对（`真实 X 被认成 Y`）：_待填_

---

## 卷积为什么更适合图像

1. **局部连接**
   MLP 的第一层是 `Linear(784, 512)`，每个神经元都连着全部 784 个像素，参数量 40 万；CNN 的 `Conv2d(1, 32, 3)` 只看 3×3 的邻域，参数量 320。
   图像的相关性本来就是局部的（相邻像素才相关），全连接是把这种先验丢掉、再从数据里重新学一遍。

2. **权值共享**
   同一个 3×3 卷积核在整张图上滑动，参数在所有位置复用。

3. **平移等变性**
   权值共享的副产品：数字平移后，卷积核的响应只是跟着移动位置，特征本身不变。MLP 在每个位置都有独立权重，简单的平移也有完全不同的输入模式。
---

## 本目录文件

| 文件 | 说明 |
| --- | --- |
| `README.md` | 本文档 |
| `src/model.py` | CNN 定义、参数量统计、按配置重建模型 |
| `src/train.py` | 训练与验证循环 |
| `src/infer.py` | 单张图片推理 |
| `src/compare.py` | MLP vs CNN 对比，生成对比图与数字 |
