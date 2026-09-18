# Level 0：环境管理

目标：在 Windows 上通过 WSL2 使用 Linux，用 Miniconda 建立独立、可复现、支持 GPU 加速的 Python 环境

---

## 配置

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11 家庭中文版 |
| WSL / 发行版 | WSL2 / Ubuntu 24.04.4 LTS |
| 内核 | 6.18.33.2-microsoft-standard-WSL2 |
| CPU | i9-14900HX（32 逻辑线程） |
| 内存 | 7.6 GB（WSL 内可见） |
| GPU / 显存 | RTX 5060 Laptop GPU / 8151 MiB |
| NVIDIA 驱动 | 582.05 |
| Conda / 环境名 | 26.7.1 / `dian-ai` |
| Python | 3.11.9 |
| PyTorch | 2.11.0+cu128（CUDA Runtime 12.8） |

---

## 安装步骤

在 WSL 终端中执行。

1. 安装基础工具与 Miniconda

```bash
sudo apt update
sudo apt install -y curl ca-certificates git build-essential

cd /tmp
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

询问是否初始化 shell 时 `yes`

2. 创建环境并安装 GPU 版 PyTorch

```bash
conda create -n dian-ai python=3.11.9 pip -y
conda activate dian-ai
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

---

## 验证

`check_env.py` 不只看版本号，还会在 GPU 上实际做一次矩阵运算并把结果取回 CPU

```bash
conda activate dian-ai
python level0_environment/check_env.py
```

实际输出：

```text
Python: 3.11.9
PyTorch: 2.11.0+cu128
PyTorch CUDA runtime: 12.8
CUDA available: True
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
GPU count: 1
GPU calculation result: 9.542873885948211e-05
```

---

## 遇到的问题

1. 必须用 cu128

**原因**：RTX 5060 是 Blackwell 架构（`sm_120`），cu118 / cu121 的预编译包不含对应 kernel。
**现象**：`torch.cuda.is_available()` 为 `True`，但一执行卷积就报 `no kernel image is available for execution on the device`。

2. 项目不放 `/mnt/c`

WSL 访问 `/mnt/c` 要经过 9P 协议转换，大量小文件读写明显变慢，而训练时 DataLoader 正是反复读小文件。改成 `/home/lenovo/projects/dian-2026-ai-csl`。

---

## 复现环境

方式一，用 Conda 环境文件：

```bash
conda env create -f environment.yml
conda activate dian-ai
```

方式二，用 pip 依赖文件：

```bash
conda create -n dian-ai python=3.11.9 pip -y
conda activate dian-ai
python -m pip install -r requirements.txt
```

---

## 本目录文件

| 文件 | 说明 |
|---|---|
| `README.md` | 本文档 |
| `check_env.py` | 环境验证脚本 |
