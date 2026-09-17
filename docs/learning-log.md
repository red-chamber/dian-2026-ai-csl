# 学习记录

> 本文件是本项目**唯一**的学习文档（对应题目要求 2：在学习过程中维护一个学习文档，记录工程开发中遇到的各种问题或思路）。
>
> 约定：
> - 按时间顺序追加，**新记录写在文件末尾**，不删除旧内容；
> - 每个 Level 结束补一条，中途遇到的问题随时插入；
> - 只记录真实发生的事，错误信息保留原文，不写「运行失败」这类无法复查的描述。
>
> 2026-09-17：Level 0 的内容原先单独放在 `level0_environment/level0_learning_document.md`，
> 现并入本文件（方案 A：全项目只保留这一份学习文档），原文件已删除，避免同一份内容散落两处。

---

## 2026-09-15 ~ 09-16 ｜ Level 0：环境管理

### 今日目标

按题目要求，在 Windows 上通过 WSL2 使用 Linux，安装 Miniconda 做环境管理，建立一个独立、可复现、
支持 GPU 加速的 PyTorch 环境，并用 `torch.cuda.is_available()` 验证它真的可用。

### 学到的概念

**1. Conda 的作用**

最初我只总结出三条，理解加深后补成四条：

1. **创建不同的环境，里面可以使用不同的 Python 版本。** 本机系统 Python 是 3.12.3，题目推荐 3.11.9，靠环境隔离才能让两个版本同时存在。
2. **不同项目的依赖不会冲突。** A 项目要 numpy 1.x、B 项目要 numpy 2.x 时，装在不同环境里互不影响，不会出现「装了 B 的依赖把 A 跑挂」。
3. **可以让他人方便地复现项目的环境。** `conda env export` 导出 `environment.yml`，别人一条 `conda env create` 就能建出等价环境。
4. **环境搞坏了可以直接删掉重建**，不会污染整个系统。

「可复现」的本质是：把自己电脑上能跑的状态，变成别人也能一键还原的**声明式描述**，而不是一堆「我记得当时敲过什么」。

另外记住一条：**不要把 Conda 环境目录本身提交到 Git**。它有几万个文件、几百 MB，而且里面全是本机绝对路径，
提交上去既没意义又会让仓库膨胀。Git 里只放 `environment.yml` / `requirements.txt` 这类描述文件。

**2. PyTorch / CUDA Runtime / NVIDIA 驱动是三层关系**

一开始我以为「CUDA 版本」只有一个，看到 `nvidia-smi` 显示 13.0 而 `torch.version.cuda` 显示 12.8 时，以为装错了。实际是：

- **NVIDIA 驱动**（最底层）：决定这台机器**能支持到**哪个 CUDA 版本，`nvidia-smi` 显示的就是这个上限；
- **CUDA Runtime**（中间层）：PyTorch 自己打包进来的，`torch.version.cuda` 是它**实际编译链接**的版本；
- **PyTorch**（上层）：我们直接调用的那一层。

只要 Runtime 不高于驱动上限就是兼容的，所以 12.8 的 Runtime 跑在支持 13.0 的驱动上完全正常。

**3. 为什么 RTX 5060 必须用 cu128**

RTX 5060 Laptop 是 Blackwell 架构，计算能力 `sm_120`。较早的 CUDA 预编译包（cu118、cu121）里**没有 sm_120 的 kernel**，
装上去会出现「设备能识别、`is_available()` 也返回 `True`，但一执行卷积就报
`no kernel image is available for execution on the device`」。所以选了 `cu128` 的 wheel。

### 实际操作

在 WSL 终端里执行：

```bash
wsl -d Ubuntu-24.04
sudo apt update
sudo apt install -y curl ca-certificates git build-essential

# 安装 Miniconda（装到默认的 /home/lenovo/miniconda3，询问时选 yes 初始化 shell）
cd /tmp
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda config --set auto_activate_base false   # 免得每次开终端都自动进 base

# 建环境
conda create -n dian-ai python=3.11.9 pip -y
conda activate dian-ai

# 装 GPU 版 PyTorch
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> 说明：上面这几条是按官方流程整理的，**不是**从终端历史里逐条抄回来的（`~/.zsh_history` 里没留下 Miniconda 的安装记录），
> 但与最终环境的实际状态（Python 3.11.9 / torch 2.11.0+cu128）一致。

**另一个关键决定：项目放在 Linux 文件系统里。**

项目路径是 `/home/lenovo/projects/dian-2026-ai-csl`，而不是 `/mnt/c/...`。
WSL 访问 `/mnt/c` 要经过 9P 协议转换，大量小文件读写会明显变慢，而训练时 DataLoader 恰好就是在反复读小文件。

### 遇到的问题

**问题 1：`conda: command not found`**

- **现象**：WSL 里 `python3` 可用，但 `conda` 命令不存在。
- **原因**：Miniconda 没装，或者装了但没初始化 shell。
- **解决**：安装 Miniconda，安装脚本问 "Do you wish to update your shell profile to automatically initialize conda?" 时选 `yes`，再 `source` 对应的 rc 文件。
- **注意**：本机默认登录 shell 是 **zsh**，所以初始化代码实际写在 `~/.zshrc` 而不是 `~/.bashrc`。找不到 `conda` 时，先确认初始化写进了哪个 rc 文件。

**问题 2：Git 报 `Please tell me who you are`**

- **原因**：没配置提交身份。
- **解决**：

```bash
git config --global user.name "red-chamber"
git config --global user.email "113877110@qq.com"
git config --global init.defaultBranch main
```

**问题 3：没有 SSH 密钥，推送没有权限**

- **原因**：本地从来没生成过 SSH 密钥，GitHub 认不出推送者身份。
- **解决**：

```bash
ssh-keygen -t ed25519 -C "113877110@qq.com"
cat ~/.ssh/id_ed25519.pub        # 把公钥内容贴到 GitHub 的 SSH keys 里
ssh -T git@github.com            # 验证是否通了
```

另外，本机 `~/.ssh/config` 里把 GitHub 的连接指到了 443 端口，因为网络屏蔽了 22 端口：

```text
Host github.com
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/id_ed25519
```

> `id_ed25519.pub` 是公钥，可以公开；`id_ed25519` 是私钥，**绝对不能**上传或提交到 Git。

**问题 4（容易误判，不是错误）：环境放错文件系统 / 重复初始化**

最初在 Windows 侧的 `D:\DianProjects` 下试过一次 `git init`，后来整个废弃了——那个目录里只有 `.git`，
没有任何工作区文件和提交，远程名还拼成了 `origain`，指向的仓库在 GitHub 上根本不存在。
真正的项目是在 WSL 里重建的，两者用的是**两套不同的 Git**（Windows 侧 `D:\Git\cmd\git.exe` 与 WSL 内的 Git 2.43.0），
身份、密钥、行尾设置都是分开的。

**教训**：先想清楚项目放在哪个文件系统、用哪一套 Git，再动手 `git init`。
判断一个仓库是不是废弃的，看两点就够：有没有提交（`.git` 里的 objects）、远程仓库是否真实存在。

### 验证方法

光看版本号和 `is_available()` 不够——必须让 GPU **真的算一次**。
写了 [`check_env.py`](../level0_environment/check_env.py)：除了打印版本，还会在 GPU 上建一个 1024×1024 随机矩阵、求均值、再把结果取回 CPU。

实际运行输出：

```text
(dian-ai) lenovo@Legion-LAPTOP-CSL:~/projects/dian-2026-ai-csl % python level0_environment/check_env.py
Python: 3.11.9
PyTorch: 2.11.0+cu128
PyTorch CUDA runtime: 12.8
CUDA available: True
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
GPU count: 1
GPU calculation result: 0.001543294871225953
```

`CUDA available: True` 只说明 PyTorch 认得这块卡；最后一行数值能正常算出来并取回 CPU，才说明**数据通路是真的通的**。
环境配置成功。

### 尚未理解的问题

- 「检查通过但一跑就炸」的情况（`is_available()` 为 `True`，kernel 却缺失），除了 cu128 这类经验规则，有没有通用的提前检测办法？
  比如用 `torch.cuda.get_device_capability()` 拿到算力，再和 wheel 支持的 `sm_` 列表做比对。
- conda 装的包和 pip 装的包混在一起时，`conda list` 与 `pip list` 的结果不完全一致，长期维护该以哪个为准。
- `environment.yml` 里 `pip:` 子段和 conda 依赖的优先关系，还没完全搞清楚。

### 参考资料

- Miniconda 官方安装文档：https://docs.conda.io/projects/miniconda/en/latest/
- PyTorch 官方安装指引（含 cu128 wheel 索引）：https://pytorch.org/get-started/locally/
- PyTorch 历史版本与 CUDA 对应关系：https://pytorch.org/get-started/previous-versions/

### AI 使用记录

- **用 AI 做了什么**：让 AI 帮忙起草了 `check_env.py`，以及 Level 0 的环境说明 `README.md`。
- **哪些内容经过验证**：脚本能跑出上面那段真实输出；`torch.cuda.is_available()` 为 `True` 是自己实际执行确认的，不是照抄 AI 的说法。
- **哪些内容要打折看**：
  - AI 写的「安装步骤」是按官方流程**重构**出来的，不是真实命令历史，顺序未必和我当时敲的完全一致。
  - AI 生成的文档里，出现过一段写给 AI 自己的指令（「每个 Level 的 README 单独记录：……」）被误留在根 README 末尾，属于该清理的噪声。
- **经验**：AI 适合用来起草脚本和文档结构；但**凡是涉及「我做过什么」的事实性描述，必须自己核对**——
  AI 会按「看起来合理」的方式补全，而不是按真实历史还原。

---

<!-- 后续 Level 的记录追加在这条横线下面，保持时间顺序 -->
