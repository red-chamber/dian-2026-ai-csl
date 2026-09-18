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
