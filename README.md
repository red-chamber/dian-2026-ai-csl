# Dian 2026 团队秋招算法题

从神经网络基础出发，逐步完成 **MLP → CNN → 经典网络 → U-Net** 的学习与实践，最终实现一个完整项目：
**输入带手写内容的试卷/作业图片，输出擦除手写内容后的干净图片。**

每个 Level 都保留代码、实验记录与 README，全程用 Git 管理版本。

## 项目简介

题目分 5 个 Level 递进，前 4 个 Level 是打基础，Level 4 是最终目标：

| Level | 目标一句话 | 关键产物 | 验收标准 |
|---|---|---|---|
| **0 环境管理** | 用 WSL2 + Miniconda 建立独立、可复现、能用 GPU 的 Python 环境 | `level0_environment/` | `torch.cuda.is_available()` 返回 `True` |
| **1 MLP** | 把 MNIST 图片展平后喂给 MLP，理解前向计算、损失与反向传播的完整流程 | `level1_mlp/` | 测试集准确率 ≥ 90%（不数据泄露）、Loss 曲线可视化、README 记录网络结构与超参数 |
| **2 CNN** | 把 MLP 换成 CNN，通过对照实验理解卷积为什么更适合图像 | `level2_cnn/` | CNN 测试集准确率约 96%，从准确率/参数量/收敛速度/错误样本四个角度与 MLP 对比 |
| **3 经典网络** | 理解 AlexNet → VGG → ResNet 的演进，并掌握 U-Net 的编码器-解码器与跳跃连接 | `level3_classic_networks/` | 在已有代码基础上分别实现 AlexNet、ResNet |
| **4 U-Net 擦除** | 用 U-Net 做手写内容擦除：保留印刷文字/表格/题目结构，擦掉手写部分 | `level4_unet/` | 输出 PSNR/SSIM 演化曲线；结果不出现全白/全黑；记录训练时长与费用（预算约 30 元） |

> 当前进度：Level 0 至 Level 4 全部完成。Level 4 在远端 RTX 4090D 上训练 60 轮，
> test PSNR 23.74 dB / SSIM 0.9584。
> 详见 [`docs/learning-log.md`](docs/learning-log.md)。

## 硬件与软件环境

实测配置（非模板值）：

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11 家庭中文版（Build 26200） |
| WSL | WSL2 |
| Linux 发行版 | Ubuntu 24.04.4 LTS (Noble Numbat) |
| 内核 | 6.18.33.2-microsoft-standard-WSL2 |
| CPU | Intel(R) Core(TM) i9-14900HX（32 逻辑线程） |
| 内存 | 15.7 GB |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| 显存 | 8151 MiB（约 8 GB） |
| NVIDIA 驱动 | 582.05 |
| WSL 内 Git | 2.43.0 |
| Miniconda 安装位置 | `/home/lenovo/miniconda3` |
| Conda 环境名 | `dian-ai` |
| Python | 3.11.9 |
| PyTorch | 2.11.0+cu128（CUDA Runtime 12.8） |

> **为什么是 cu128**：RTX 5060 Laptop 是 Blackwell 架构（`sm_120`），cu118 / cu121 的预编译包里没有对应 kernel，
> 会出现「`is_available()` 为 `True` 但一跑卷积就报 `no kernel image is available`」。原因见图 [Level 0 README](level0_environment/README.md)。

项目放在 WSL 的 Linux 文件系统内（`/home/lenovo/projects/dian-2026-ai-csl`），**不放在 `/mnt/c`**：
跨文件系统访问要经过 9P 协议转换，小文件读写明显变慢，而训练时 DataLoader 正是大量小文件读取。

## 仓库结构

```text
dian-2026-ai-csl/
├── README.md                       # 本文件：项目总览
├── environment.yml                 # Conda 环境导出（含 pip 子段，含 torch cu128）
├── requirements.txt                # pip 依赖清单
├── .gitignore
├── docs/
│   ├── learning-log.md             # 【唯一】学习文档，按时间追加
│   └── experiment-log.md           # 实验记录模板
├── level0_environment/             # ✅ 已完成
│   ├── README.md                   #   环境说明与验收记录
│   └── check_env.py                #   环境验证脚本
├── common/                         # 三个 Level 共用的训练基础设施
│   ├── utils.py                    #   路径常量、随机种子、参数量统计、指标导出
│   ├── data.py                     #   数据集注册表（MNIST / Fashion-MNIST）与 DataLoader
│   ├── engine.py                   #   训练循环、评估、完整训练流程
│   ├── checkpoint.py               #   权重的存与读（按 model_config 重建模型）
│   └── plots.py                    #   通用绘图（收敛曲线）
├── level1_mlp/                     # ✅ 已完成（MLP on MNIST，98.16%）
│   ├── README.md                   #   网络结构、超参数与实验结果
│   └── src/
│       ├── model.py                #   MLP 定义、按配置重建模型
│       ├── train.py                #   命令行接口 + 训练（流程来自 common/）
│       └── infer.py                #   单张图片推理
├── level2_cnn/                     # ✅ 已完成（CNN on MNIST，99.06%）
│   ├── README.md                   #   网络结构、超参数与对比方法
│   └── src/
│       ├── model.py                #   CNN 定义、按配置重建模型
│       ├── train.py                #   命令行接口 + 训练（只比 Level 1 多两个结构参数）
│       ├── infer.py                #   单张图片推理
│       └── compare.py              #   MLP vs CNN 四角度对比
├── level3_classic_networks/        # ✅ 已完成（AlexNet 91.42% / ResNet-18 92.63% / PlainNet-18 92.76%）
│   ├── README.md                   #   两个网络的结构、参数量与演进主线
│   └── src/
│       ├── model.py                #   AlexNet、ResNet18（含残差消融开关）
│       ├── train.py                #   命令行接口 + 训练
│       ├── infer.py                #   单张图片推理
│       └── compare.py              #   经典网络对比
├── level4_unet/                    # ✅ 已完成（test PSNR 23.74 dB / SSIM 0.9584）
│   ├── README.md                   #   数据集、U-Net 结构、损失与指标、验收对照
│   ├── configs/README.md           #   为什么用命令行参数而不是配置文件
│   └── src/
│       ├── data.py                 #   配对数据集、尺寸处理、按日期划分
│       ├── model.py                #   U-Net
│       ├── losses.py               #   L1 / MSE / L1+梯度
│       ├── metrics.py              #   PSNR / SSIM 与退化检查
│       ├── predict.py              #   尺寸补齐、分块推理、整集评估
│       ├── viz.py                  #   训练曲线与四列对比图
│       ├── engine.py               #   训练循环（断点续训、费用估算）
│       ├── train.py                #   训练入口
│       ├── infer.py                #   单张 / 批量推理
│       └── evaluate.py             #   测试集评估与验收检查
├── data/
│   ├── raw/                        # 原始数据（.gitignore 已忽略）
│   ├── processed/                  # 处理后数据（忽略）
│   └── splits/                     # 训练/验证/测试划分
├── checkpoints/                    # 模型权重（忽略）
└── reports/
    ├── figures/                    # Loss / PSNR / SSIM 曲线
    └── samples/                    # 推理结果对比图
```

> **注意**：`data/raw` 等目录目前是空的，**Git 不跟踪空目录**，克隆下来不会有这些空壳。
> 往里放进第一个文件后，目录才会真正进入版本管理。
>
> 学习文档只保留 `docs/learning-log.md` 这一份（题目要求 2：在学习过程中维护一个学习文档），
> 各 Level 的过程记录都追加到里面，不再每个 Level 各写一份。
>
> `common/` 是三个 Level 共用的训练基础设施：数据加载、训练循环、画图、指标导出都只写一遍。
> 各 Level 的 `train.py` 只保留命令行接口、建模型和实验摘要，所以在 `python level1_mlp/src/train.py`
> 这样直接运行时，脚本会先把仓库根目录补进 `sys.path` 才能 import 到 `common`。

## 环境安装与复现

### 方式一：使用 Conda 环境文件（推荐）

```bash
conda env create -f environment.yml
conda activate dian-ai
python level0_environment/check_env.py
```

### 方式二：使用 pip 依赖文件

```bash
conda create -n dian-ai python=3.11.9 pip -y
conda activate dian-ai
python -m pip install -r requirements.txt
```

### 方式三：从零手动安装

```bash
sudo apt update && sudo apt install -y curl ca-certificates git build-essential

# 安装 Miniconda（装完 source 对应 rc 文件；注意本机默认 shell 是 zsh，配置写在 ~/.zshrc）
cd /tmp && curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

conda create -n dian-ai python=3.11.9 pip -y
conda activate dian-ai
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 验证安装

```bash
conda activate dian-ai
python level0_environment/check_env.py
```

预期输出：

```text
Python: 3.11.9
PyTorch: 2.11.0+cu128
PyTorch CUDA runtime: 12.8
CUDA available: True
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
GPU count: 1
GPU calculation result: 0.001543294871225953
```

脚本不只看 `is_available()`，还会在 GPU 上实际做一次矩阵运算并把结果取回 CPU——只有这一步成功，才能证明 kernel 真的能跑。

完整的问题排查记录（`conda: command not found`、Git 身份、SSH 端口、`nvidia-smi` 与 `torch.version.cuda` 版本差异等）见
[`level0_environment/README.md`](level0_environment/README.md) 与 [`docs/learning-log.md`](docs/learning-log.md)。

## 数据集获取与目录放置

Level 4 数据集由招新方提供（见招新群的公告中的下载方式，本项目不在仓库内公开传播该链接）。

下载后解压到：

```text
data/raw/          # 原始图片
data/processed/    # 清洗后的数据
data/splits/       # 训练/验证/测试集划分文件
```

`data/raw/` 与 `data/processed/` 已在 `.gitignore` 中忽略，**不要把数据集提交进 Git**。
MNIST / Fashion-MNIST（Level 1–2）由 `torchvision.datasets` 自动下载，无需手动准备。

Level 4 的数据集是配对数据，解压后是三个日期目录，每个下面各有 input/output 两个子目录，
两边文件名一一对应（input 带手写、output 干净）：

```text
<data_root>/
├── 20250211/dataset/{input,output}/     812 对
├── 20250212/dataset/{input,output}/     900 对
└── 20250213/dataset/{input,output}/     700 对
```

共 2412 对。**不需要复制进仓库**，训练时用 `--data-root` 指向它即可：

```bash
python level4_unet/src/train.py --data-root /path/to/dataset
```

细节（格式混杂、尺寸差异、按日期划分的理由）见 [`level4_unet/README.md`](level4_unet/README.md)。

## Level 目标

- **Level 1｜MLP 完成 MNIST 手写数字识别**：把 MNIST 图片展平后输入 MLP，掌握 Tensor / Dataset / DataLoader / 自动求导、
  `nn.Module` 与优化器、训练与验证循环、权重保存加载与单张图片推理；测试集准确率 ≥ 90%。
- **Level 2｜CNN 完成 MNIST 并与 MLP 对比**：复用 Level 1 的数据与训练流程，训练 CNN 版本分类器，
  从准确率、参数量、收敛速度、错误样本四个角度对比，解释 CNN 为什么更适合图像任务；准确率约 96%。
- **Level 3｜经典网络 AlexNet、VGG 与 U-Net**：理解深层 CNN、ReLU、Dropout、数据增强、小卷积核堆叠与模块化、
  残差跳跃，以及 U-Net 的编码器-解码器、上下采样和跳跃连接；实现 AlexNet 与 ResNet。
- **Level 4｜基于 U-Net 的手写笔记擦除**：完成数据清洗、标注规范与数据集划分、文档图像增强、损失函数选择（MSE / L1 / L2 或组合），
  输出 Loss 与 PSNR / SSIM 演化曲线，控制服务器成本（约 30 元），并思考 GAN / Diffusion 等改进方向。

## 运行训练

环境准备见[环境安装与复现](#环境安装与复现)，需先 `conda activate dian-ai`。

```bash
# Level 1：MLP on MNIST（默认参数即为实验 01 的配置，可直接复现）
python level1_mlp/src/train.py

# 覆盖超参数
python level1_mlp/src/train.py --epochs 15 --hidden-sizes 512 256 128 --dropout 0.3 --tag mlp_3layer

# Level 2：CNN on MNIST（默认超参数与 Level 1 一致，保证对比只体现结构差异）
python level2_cnn/src/train.py

# 训练完后做 MLP vs CNN 的四角度对比（不重新训练，只读各自的 metrics json）
python level2_cnn/src/compare.py

# Level 3：经典网络 on Fashion-MNIST（两个模型要分别训练）
python level3_classic_networks/src/train.py --model alexnet
python level3_classic_networks/src/train.py --model resnet
python level3_classic_networks/src/compare.py

# Level 3 消融：去掉残差连接，参数量不变，看深了会怎样
python level3_classic_networks/src/train.py --model resnet --no-residual --tag resnet_plain

# Level 4：U-Net 手写内容擦除（--data-root 指向数据集，无需复制进仓库）
python level4_unet/src/train.py --data-root /path/to/dataset --epochs 60 --loss l1 --tag unet_l1

# 云上被中断后续训（恢复模型、优化器与 epoch，并估算费用）
python level4_unet/src/train.py --data-root /path/to/dataset --epochs 60 \
    --tag unet_l1 --resume --gpu-hourly-cost 1.5

# 测试集评估：输出 PSNR / SSIM、检查有没有全白/全黑，并给最差几张的对比图
python level4_unet/src/evaluate.py --checkpoint checkpoints/unet_l1_best.pt --worst 4

# 单张 / 批量推理，输出擦除后的干净图
python level4_unet/src/infer.py --checkpoint checkpoints/unet_l1_best.pt --input photo.jpg
```

训练产出（三个 Level 一致）：

| 产物 | 路径 |
|---|---|
| 最优权重（按验证准确率保存） | `checkpoints/<tag>_best.pt` |
| Loss / Accuracy 曲线 | `reports/figures/<tag>_curves.png` |
| 完整指标与逐轮 history | `reports/metrics/<tag>.json` |

训练结束会在终端打印一段实验摘要，可直接填进 [`docs/experiment-log.md`](docs/experiment-log.md)。

## 单张图片推理

`--index` 与 `--image` 互斥，必须二选一。

```bash
# Level 1：取测试集第 0 张
python level1_mlp/src/infer.py --index 0

# 自己的图片，png/jpg，白底黑字或黑底白字均可
python level1_mlp/src/infer.py --image path/to/digit.png

# Level 2：把默认权重换成 CNN 的
python level2_cnn/src/infer.py --index 0

# Level 3：--checkpoint 指到哪个模型就跑哪个（AlexNet / ResNet 自动识别）
python level3_classic_networks/src/infer.py --index 0
python level3_classic_networks/src/infer.py --checkpoint checkpoints/alexnet_fashionmnist_best.pt --index 0
```

终端打印 10 个类别的概率，同时把「原图 + 概率条形图」存到 `reports/samples/`。
Level 3 用的是 Fashion-MNIST，标签是 10 类衣物名而不是数字，图上显示的就是类别名。

## 实验结果

### Level 1：MLP on MNIST

| 项目 | 值 |
|---|---|
| **测试集准确率** | **98.16%**（验收要求 ≥ 90% ✅） |
| 参数量 | 535,818 |
| 最优 epoch | 第 7 轮（验证准确率 97.98%） |
| 训练时长 | 37.58 s |
| GPU 显存峰值 | 28.9 MB |

曲线见 `reports/figures/mlp_mnist_curves.png`，逐项记录见 [`docs/experiment-log.md`](docs/experiment-log.md)，网络结构与超参数见 [`level1_mlp/README.md`](level1_mlp/README.md)。

### Level 2：CNN on MNIST

| 项目 | MLP | CNN |
|---|---|---|
| 参数量 | 535,818 | 119,530（MLP 的 22.3%） |
| 测试集准确率 | 98.16% | **99.06%**（验收要求 ≈96% ✅） |
| 最优 epoch | 第 7 轮（97.98%） | 第 7 轮（98.85%） |
| 第 1 轮验证准确率 | 95.45% | 96.40% |
| 训练时长 | 37.58 s | 35.6 s（10 轮） |
| GPU 显存峰值 | 28.9 MB | 78.7 MB |

CNN 用约五分之一的参数量把测试准确率提高了 0.90 个百分点，并且第 1 轮就越过 96%
（MLP 需要 2 轮）。错误样本上「只有 MLP 错」136 张而「只有 CNN 错」46 张，说明提升来自
CNN 净多救回的 90 张图。四角度完整对比见 [`level2_cnn/README.md`](level2_cnn/README.md)，
对比曲线、混淆矩阵、错误样本网格分别在 `reports/figures/mlp_vs_cnn_curves.png`、
`reports/figures/mlp_vs_cnn_confusion.png`、`reports/samples/mlp_vs_cnn_errors.png`。

### Level 3：AlexNet / ResNet on Fashion-MNIST

| 模型 | 参数量 | 测试准确率 | 最优轮次 | 每轮耗时 |
|---|---|---|---|---|
| AlexNet | 5,338,314 | 91.42% | 第 14 轮 | 7.2 s |
| ResNet-18 | 11,172,810 | 92.63% | 第 19 轮 | 22.2 s |
| PlainNet-18（无残差消融） | 11,172,810 | 92.76% | 第 9 轮 | 22.0 s |

三个模型都在 20 轮内出现明显过拟合（ResNet-18 第 20 轮训练准确率 99.46%，验证准确率最高
只有 93.30%）。ResNet-18 收敛更快（第 4 轮即达 92% 验证准确率，AlexNet 全程未达到 92%），
但每轮耗时是 AlexNet 的 3.1 倍。

残差消融给出了**与预期相反**的结果：去掉跨层连接后测试准确率不降反升
（92.76% vs 92.63%），测试损失明显更低（0.2273 vs 0.3767）。18 层对 28×28 的输入而言
还不够深，退化问题没有出现，而 BatchNorm 已承担了稳定梯度的主要工作；
加上单次运行的差异可能落在噪声内，因此本项目**没有复现出残差连接的优势**。
分析见 [`level3_classic_networks/README.md`](level3_classic_networks/README.md)。

### Level 4：U-Net 手写内容擦除 on 配对文档数据

远端 RTX 4090D 训练，L1 损失，60 轮中最优在第 36 轮。数据集结构、U-Net、损失与指标设计、
验收标准逐条对照见 [`level4_unet/README.md`](level4_unet/README.md)。

| 指标 | 值 |
|---|---|
| 模型 / 参数量 | U-Net base_channels=64 / 31,036,481 |
| 验证 PSNR / SSIM（最优） | 22.00 dB / 0.9409 |
| 测试集 PSNR | 23.74 dB（最差 8.66 / 最好 54.52） |
| 测试集 SSIM | 0.9584（最差 0.5225 / 最好 0.9984） |
| 全白/全黑检查 | 验证 60 轮 0 次退化；test 1/630 近白，该张真值本身即空白页、预测 PSNR 全集最高，非失败 |
| 训练时长 / 预估费用 | 0.65 小时（39.0 分钟）/ 大约30元 |

另做了损失函数对照，三组超参与数据划分完全一致，只改 `--loss`：

| 损失 | 验证 PSNR / SSIM（最优） | 测试 PSNR | 测试 SSIM |
|---|---|---|---|
| L1（默认，主实验） | 22.00 dB / 0.9409 | 23.74 dB | 0.9584 |
| MSE | 23.12 dB / 0.9255 | 24.86 dB | 0.9493 |
| L1 + 梯度 | 22.18 dB / 0.9508 | 24.31 dB | 0.9620 |

MSE 的 PSNR 最高而 SSIM 最低，L1+梯度的 SSIM 最高 —— 与「PSNR 天然偏向 MSE、边缘约束改善结构相似度」的预期一致。
差距都在 1.2 dB / 0.013 以内且未做多种子，不作为选型结论，详见 [`docs/experiment-log.md`](docs/experiment-log.md) 实验 05。

## 已知问题

- 空目录（`data/raw`、`level4_unet/src` 等）不会被 Git 跟踪，克隆后需手动创建或靠首次提交带入。
- `data/` 下除 `raw/`、`processed/` 之外的路径（如 `data/foo.csv`）**不在** `.gitignore` 忽略范围内，提交前需留意。
- 各 Level 的 `src/` 下都有名为 `model.py` 的文件，`from model import ...` 之所以不冲突，是因为脚本每次都在独立进程里、按 `python levelN/src/xxx.py` 的方式运行（此时脚本所在目录会排在 `sys.path` 最前）。如果要在同一个进程里同时用到两个 Level 的模型，必须像 `level2_cnn/src/compare.py` 那样用 `importlib` 按文件路径加载，不能直接 `import model`。

## AI 使用说明

题目允许并鼓励使用 AI，但要求理解代码而非复制。本项目的原则：

- AI 用于起草脚本、文档结构与排查思路；**代码必须能逐行讲清数据怎么流动**。
- 凡是「我做过什么」的事实性描述（安装步骤、运行输出、报错信息），一律以实际执行结果为准，不采信 AI 的合理推测。
- 记录哪些 AI 建议经实验验证、哪些不适合本项目，写在 [`docs/learning-log.md`](docs/learning-log.md) 的「AI 使用记录」小节。

## 参考资料

- Miniconda 文档：https://docs.conda.io/projects/miniconda/en/latest/
- PyTorch 安装指引：https://pytorch.org/get-started/locally/
- PyTorch 历史版本与 CUDA 对应：https://pytorch.org/get-started/previous-versions/
- U-Net 原论文：Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015. https://arxiv.org/abs/1505.04597
