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