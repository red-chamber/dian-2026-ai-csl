# Level 1：MLP 完成 MNIST 手写数字识别

目标：把 MNIST 图片展平后喂给 MLP，跑通「数据加载 → 前向计算 → 损失 → 反向传播 → 参数更新」的完整流程，测试集准确率 ≥ 90%

---

## 网络结构

```
输入 (N, 1, 28, 28)
  │
  ├─ Flatten ────────────────────────→ (N, 784)          28×28 = 784
  │
  ├─ Linear(784, 512) + ReLU + Dropout → (N, 512)
  ├─ Linear(512, 256) + ReLU + Dropout → (N, 256)
  ├─ Linear(256, 10) ─────────────────→ (N, 10)   logits
  │
输出 每个类别的原始打分（未过 softmax）
```

| 层 | 配置 | 参数量 |
|---|---|---|
| Flatten | `nn.Flatten()` | 0 |
| 隐藏层 1 | `Linear(784, 512)` + ReLU + Dropout(0.2) | 784×512 + 512 = 401,920 |
| 隐藏层 2 | `Linear(512, 256)` + ReLU + Dropout(0.2) | 512×256 + 256 = 131,328 |
| 输出层 | `Linear(256, 10)` | 256×10 + 10 = 2,570 |
| **合计** | | **535,818** |

两点说明：

- 隐藏层用 `nn.ModuleList` 而非 `nn.Sequential`，是为了在 `forward` 里显式写出数据流，而不是藏成一行黑盒。
- 输出层**不加 softmax**。损失函数是 `nn.CrossEntropyLoss`，它内部已经等价于 `log_softmax + NLLLoss`，再加一次就是重复施加。

---

## 超参数

| 项目 | 值 |
|---|---|
| 数据集 | MNIST（train 54,000 / val 6,000 / test 官方 10,000） |
| 划分方式 | 从官方训练集 60,000 张中切 10% 作验证集，测试集不参与任何决策 |
| 输入尺寸 | 1 × 28 × 28（展平为 784） |
| batch size | 128 |
| 优化器 | Adam，lr = 1e-3，weight_decay = 0 |
| 损失函数 | CrossEntropyLoss |
| epochs | 10 |
| dropout | 0.2 |
| 权重初始化 | Kaiming Normal（`nonlinearity="relu"`），偏置置零 |
| 随机种子 | 42 |
| 设备 | CUDA（RTX 5060 Laptop GPU） |

**数据泄露检查**：验证集从训练集切出，与测试集无交集；模型选择只看验证准确率，测试集只在训练全部结束后评估一次。

---

## 运行

环境准备见仓库根 README，需先 `conda activate dian-ai`。

训练（默认参数即为本次实验配置，可直接复现）：

```bash
python level1_mlp/src/train.py
```

常用覆盖项：

```bash
python level1_mlp/src/train.py --epochs 15 --hidden-sizes 512 256 128 --dropout 0.3 --tag mlp_3layer
```

单张图片推理（二选一，`--index` 与 `--image` 互斥且必选其一）：

```bash
# 取测试集第 0 张
python level1_mlp/src/infer.py --index 0

# 自己的图片（png/jpg，白底黑字或黑底白字均可）
python level1_mlp/src/infer.py --image path/to/digit.png
```

推理会把原图和 10 个类别的概率条形图存到 `reports/samples/`。

---

## 实验结果

| 项目 | 值 |
|---|---|
| **测试集准确率** | **98.16%**（验收要求 ≥ 90% ✅） |
| 测试集损失 | 0.0672 |
| 最优 epoch | 第 7 轮，验证准确率 97.98% |
| 第 10 轮训练准确率 | 98.82% |
| 参数量 | 535,818 |
| 训练时长 | 37.58 s |
| GPU 显存峰值 | 28.9 MB |

Loss / Accuracy 曲线：`reports/figures/mlp_mnist_curves.png`
完整指标：`reports/metrics/mlp_mnist.json`
权重文件：`checkpoints/mlp_mnist_best.pt`（已在 `.gitignore` 中忽略）

### 结论

- 训练准确率从第 1 轮的 91.1% 涨到第 10 轮的 98.8%，但**验证准确率第 7 轮（97.98%）之后不再上升**；验证损失则从第 5 轮的 0.0808 回升到第 10 轮的 0.0905 —— 典型的过拟合迹象。训练脚本按验证准确率保存最优权重，因此最终用的是第 7 轮的模型而不是第 10 轮的。
- 单层就达到 91% 说明 **MNIST 用 MLP 已经能解**，代价是参数量高达 53.6 万（几乎全在第一层 784×512）。
- MLP 的第一步是 `Flatten`，**图像被压成一维向量后二维空间结构就丢了** —— 相邻像素在向量里不再相邻，模型只能靠全连接层硬记位置关系。这是 Level 2 要换 CNN 的核心动机。

---

## 遇到的问题

1. 自己的图片预测不准

**原因**：网上找的手写数字图多是白底黑字，而 MNIST 是黑底白字，两者像素分布正好相反，模型没见过这种输入。
**解决**：读入后求全图像素均值，大于 127 判定为白底，做 `255 - x` 反色（`load_image_from_file`）。

2. 推理时必须关掉 Dropout

**现象**：同一张图连续推理两次，输出的概率不一样。
**原因**：`model.eval()` 漏写，Dropout 仍在随机丢神经元。
**解决**：加载权重后立即 `model.eval()`，并用 `@torch.no_grad()` 关掉计算图。

3. `--index 0` 被判成「没给参数」

**原因**：`0` 在 Python 中是 falsy，`if args.index:` 会把 `--index 0` 误判成未提供。
**解决**：写成 `if args.index is not None:`。

---

## 本目录文件

| 文件 | 说明 |
|---|---|
| `README.md` | 本文档 |
| `src/model.py` | MLP 定义、参数量统计、按配置重建模型 |
| `src/train.py` | 训练与验证循环、保存最优权重、画曲线、导出实验指标 |
| `src/infer.py` | 单张图片推理（测试集索引或自定义图片） |
