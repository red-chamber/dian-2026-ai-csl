# Level 0：环境管理

> 目标：在 Windows 上通过 WSL2 使用 Linux，用 Miniconda 建立一个独立、可复现、支持 GPU 加速的 Python 环境。

## 1. 学习目标

- 在 Windows 上通过 WSL2 获得 Linux 命令行环境
- 安装 Miniconda，理解 Conda 解决了什么问题
- 创建 Python 3.11.9 的独立环境，与系统 Python 隔离
- 安装 GPU 版 PyTorch，并确认 `torch.cuda.is_available()` 返回 `True`

## 2. 我的电脑配置

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

## 3. 为什么使用 WSL2

- 题目要求 WSL，同时深度学习生态（PyTorch、各种数据加载库）在 Linux 下兼容性最好，报错信息也更容易检索到。
- WSL2 是真正的 Linux 内核，不是模拟层，因此对 NVIDIA 驱动和 CUDA 的支持是原生的：Windows 侧装好显卡驱动后，WSL 内用 `nvidia-smi` 就能直接看到 GPU，不需要在 WSL 里再单独装一遍驱动。
- 文件位置很重要：项目放在 Linux 文件系统（`/home/lenovo/...`）而不是 `/mnt/c/...`。跨文件系统访问（`/mnt/c`）要经过 9P 协议转换，大量小文件读写会明显变慢，而训练时 DataLoader 恰好是大量小文件读取。

## 4. Conda 的作用

1. **隔离 Python 版本**：不同项目可以用不同的 Python 版本，互不影响。本机系统 Python 是 3.12.3，而题目建议 3.11.9，靠环境隔离才能同时存在。
2. **隔离依赖**：不同项目需要的同一个库的不同版本不会互相覆盖冲突。
3. **可复现**：环境可以导出成 `environment.yml`，别人一条命令就能建出等价环境。
4. **可重建**：环境被搞坏时，直接删掉重建，不会污染整个系统。

需要注意：**不要把 Conda 环境目录本身提交到 Git**（它包含几万个文件、几百 MB，且里面全是本机绝对路径）。Git 里只保存环境的描述文件。

## 5. 环境安装步骤

以下步骤在 WSL 终端（Ubuntu-24.04）中执行。进入 WSL：

```bash
wsl -d Ubuntu-24.04
```

### 5.1 安装基础工具

```bash
sudo apt update
sudo apt install -y curl ca-certificates git build-essential
```

### 5.2 安装 Miniconda

下载 Linux x86-64 安装脚本并执行：

```bash
cd /tmp
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

交互过程中：

1. 阅读协议按 `Enter`，输入 `yes` 接受；
2. 安装目录直接回车，使用默认的 `/home/lenovo/miniconda3`；
3. 询问是否初始化 Conda 时输入 `yes`。

让配置在当前终端立即生效：

```bash
source ~/.bashrc
conda --version
```

关闭每次开终端自动进入 `base` 环境：

```bash
conda config --set auto_activate_base false
```

> 本机默认登录 shell 是 **zsh**，因此 conda 的初始化代码实际写在 `~/.zshrc` 中。若 `conda` 命令找不到，先确认初始化写入了哪个 rc 文件。

### 5.3 创建项目环境

```bash
conda create -n dian-ai python=3.11.9 pip -y
conda activate dian-ai
python --version
which python
```

预期 `which python` 指向：

```text
/home/lenovo/miniconda3/envs/dian-ai/bin/python
```

### 5.4 安装 GPU 版 PyTorch

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> **为什么必须是 cu128 及以上**：RTX 5060 Laptop 是 Blackwell 架构，计算能力为 `sm_120`。较早的 CUDA 预编译包（如 cu118、cu121）不包含 `sm_120` 的 kernel，会出现「`torch.cuda.is_available()` 为 `True`，但一执行卷积就报 `no kernel image is available for execution on the device`」这类问题。

注意：不需要在 WSL 里额外安装完整的 CUDA Toolkit，使用 PyTorch 自带的 CUDA Runtime 即可。

## 6. PyTorch 与 CUDA 验证

验证脚本见 [`check_env.py`](./check_env.py)：

```python
import platform
import torch


def main():
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU count:", torch.cuda.device_count())

        x = torch.randn(1024, 1024, device="cuda")
        print("GPU calculation result:", x.mean().item())


if __name__ == "__main__":
    main()
```

运行：

```bash
conda activate dian-ai
python level0_environment/check_env.py
```

本机实际输出：

```text
Python: 3.11.9
PyTorch: 2.11.0+cu128
PyTorch CUDA runtime: 12.8
CUDA available: True
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
GPU count: 1
GPU calculation result: 0.001543294871225953
```

`CUDA available: True` 说明 PyTorch 已经能调用 GPU；最后一行是在 GPU 上真实做了一次矩阵运算并把结果取回 CPU，证明数据通路是通的，而不是只检测到了设备。

## 7. 遇到的问题与解决方法

### 7.1 `conda: command not found`

**现象**：WSL 中已确认 `python3` 可用，但 `conda` 命令不存在。

**原因**：Miniconda 尚未安装，或安装后没有初始化 shell。

**解决**：按 5.2 安装 Miniconda，并在安装脚本询问 "Do you wish to update your shell profile to automatically initialize conda?" 时选择 `yes`，然后 `source` 对应的 rc 文件。

### 7.2 Git 未配置提交身份

**现象**：`git commit` 报 `Please tell me who you are`。

**解决**：

```bash
git config --global user.name "red-chamber"
git config --global user.email "113877110@qq.com"
git config --global init.defaultBranch main
```

### 7.3 没有 SSH 密钥，推送没有权限

**原因**：本地从未生成过 SSH 密钥，GitHub 无法识别推送者身份。

**解决**：生成 ed25519 密钥并把公钥添加到 GitHub：

```bash
ssh-keygen -t ed25519 -C "113877110@qq.com"
cat ~/.ssh/id_ed25519.pub
ssh -T git@github.com
```

本机 `~/.ssh/config` 中把 GitHub 的 SSH 连接指向了 `ssh.github.com:443`，这样即使网络屏蔽了 22 端口也仍能推送：

```text
Host github.com
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/id_ed25519
```

> `id_ed25519.pub` 是公钥，可以公开；`id_ed25519` 是私钥，绝不能上传或提交到 Git。

### 7.4 两个容易误判的点

**（1）`nvidia-smi` 显示的 CUDA 版本和 `torch.version.cuda` 不一致。**

本机 `nvidia-smi` 显示 `CUDA Version: 13.0`，而 `torch.version.cuda` 是 `12.8`。这不是装错了：

- `nvidia-smi` 显示的是**驱动支持的最高 CUDA 版本**，即驱动能力上限；
- `torch.version.cuda` 显示的是 **PyTorch 编译时链接的 CUDA Runtime 版本**，即实际使用的版本。

只要后者不高于前者，就是兼容的。所以 12.8 的运行时跑在支持 13.0 的驱动上完全正常。

**（2）`torch.cuda.is_available()` 为 `True` 不等于 GPU 一定能算。**

如果 CUDA 版本与显卡架构不匹配，设备可以被识别、检查也通过，但真正执行 kernel 时才会报错。所以验证一定要像 `check_env.py` 那样**实际做一次 GPU 运算**，而不是只打印 `is_available()`。

## 8. 环境复现方法

方式一，使用 Conda 环境文件（推荐）：

```bash
conda env create -f environment.yml
conda activate dian-ai
python level0_environment/check_env.py
```

方式二，使用 pip 依赖文件：

```bash
conda create -n dian-ai python=3.11.9 pip -y
conda activate dian-ai
python -m pip install -r requirements.txt
```

> `environment.yml` 已通过 `conda env export --no-builds | sed '/^prefix:/d'` 导出去掉了本机绝对路径，因此在别人的机器上也能创建成功。

## 9. 学习总结

- 环境管理解决的问题是**可复现**：把自己电脑上能跑的状态，变成别人也能一键还原的声明式描述。
- PyTorch、CUDA Runtime、NVIDIA 驱动三者是分层关系：驱动在最底层提供能力上限，CUDA Runtime 是被 PyTorch 打包进来的中间层，PyTorch 是上层接口。用户通常只需要通过驱动版本去约束 CUDA 版本的选择。
- 环境验证要「跑通一次真实计算」，而不是只看版本号或设备是否可见。

## 10. 本目录内容

| 文件 | 说明 |
|---|---|
| `README.md` | 本文档，Level 0 的环境说明与验收记录 |
| `check_env.py` | 环境验证脚本，检查 Python / PyTorch / CUDA / GPU |
| `level0_learning_document.md` | 学习过程中记录的问题与思考 |
