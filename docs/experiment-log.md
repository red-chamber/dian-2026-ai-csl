# 实验记录

每次实验至少记录以下 15 项，按时间从上往下追加，最新的实验放在最上面。模板见文末。

---

## 实验 01 ｜ 2026-09-17 ｜ Level 1：MLP on MNIST

| 字段 | 值 |
|---|---|
| 实验编号与日期 | 01 / 2026-09-17 23:46 |
| Git commit ID | `871020a` |
| 模型名称 | MLP [512, 256] |
| 数据集和划分方式 | MNIST；官方训练集 60,000 张中切 10% 作验证集 → train 54,000 / val 6,000；测试集用官方 10,000 张 |
| 随机种子 | 42 |
| 输入尺寸 | 1 × 28 × 28（展平为 784） |
| batch size | 128 |
| optimizer / learning rate / epoch | Adam / 1e-3 / 10 epochs（每 epoch 422 个 step） |
| 损失函数 | CrossEntropyLoss |
| 参数量 | 535,818 |
| 测试准确率 | **98.16%**（测试损失 0.0672） |
| 训练时长 | 37.58 s |
| GPU 显存占用 | 峰值 28.9 MB |
| 结果图路径 | `reports/figures/mlp_mnist_curves.png` |

补充信息：

- dropout = 0.2，weight_decay = 0（未启用 L2 正则）
- 权重初始化：Kaiming Normal（`nonlinearity="relu"`），偏置置零
- 最优 epoch = 第 7 轮，该轮验证准确率 97.98%；第 10 轮训练准确率 98.82%
- 环境：Python 3.11.9 / PyTorch 2.11.0+cu128 / RTX 5060 Laptop GPU
- 权重文件：`checkpoints/mlp_mnist_best.pt`（已 gitignore）
- 完整指标与逐轮 history：`reports/metrics/mlp_mnist.json`

![MLP Loss / Accuracy 曲线](../reports/figures/mlp_mnist_curves.png)

### 结论和下一步修改

**结论**

1. 测试准确率 98.16%，超过 90% 的验收标准。训练准确率第 1 轮就有 91.1%，说明 MNIST 对 MLP 而言并不难。
2. 出现过拟合迹象：训练准确率持续上升到 98.8%，但验证准确率第 7 轮后不再提升；验证损失从第 5 轮的 0.0808 回升到第 10 轮的 0.0905。由于训练脚本按验证准确率保存最优权重，最终使用的是第 7 轮的模型。
3. 参数量 535,818，其中第一层 `Linear(784, 512)` 占 401,920（75%）—— 瓶颈来自「把图像展平成一维」这一步。

**下一步修改**

1. 进入 Level 2，换 CNN 做对照实验，重点比较参数量与收敛速度：卷积的局部连接和权值共享有望用远少于 53.6 万的参数达到更高准确率。
2. 若继续用 MLP，可尝试加大 dropout（0.2 → 0.3/0.5）或启用 weight_decay，观察验证损失回升是否被推迟。
3. 补充错误样本分析（哪些数字被认错、错在哪些类别之间），为 Level 2 的「错误样本对比」做准备。

---

## 记录模板

复制下面字段填写新实验：

```text
## 实验 NN ｜ YYYY-MM-DD ｜ Level X：<实验名>

- 实验编号与日期
- Git commit ID
- 模型名称
- 数据集和划分方式
- 随机种子
- 输入尺寸
- batch size
- optimizer、learning rate、epoch/step
- 损失函数
- 参数量
- 测试准确率或 PSNR/SSIM
- 训练时长
- GPU 显存占用
- 结果图路径
- 结论和下一步修改
```
