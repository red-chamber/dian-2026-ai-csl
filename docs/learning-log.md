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

想试不同学习率，如果直接改代码要在源码里改，改一次污染一次 Git 历史，还容易忘记改回。用命令行则命令本身就是实验记录，有利于复现结果和记录

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