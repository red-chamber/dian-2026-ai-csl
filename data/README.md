# data

数据目录。除本文件外整个目录在 `.gitignore` 里被忽略，所以克隆仓库后只有这一个文件。

## 目录约定

```text
data/
├── raw/         原始数据
├── processed/   清洗后的数据
└── splits/      训练 / 验证 / 测试的划分文件
```

`raw/`、`processed/`、`splits/` 目前都不存在，Git 不跟踪空目录。它们会在第一次用到时自动建立：Level 1–3 的 `build_dataloaders` 里有一句 `mkdir(parents=True, exist_ok=True)`，而这三个 Level 的 `--data-dir` 默认值就是 `data/raw`。

## 各 Level 的数据放哪

| Level | 数据集 | 放置方式 |
| --- | --- | --- |
| 1、2 | MNIST | `torchvision.datasets` 自动下载到 `data/raw`，不需要手动准备 |
| 3 | Fashion-MNIST | 同上 |
| 4 | 配对文档图（招新方提供） | 不复制进仓库，用 `--data-root` 指向解压后的目录 |

Level 4 的数据集是三个日期目录，每个下面有 `input` 和 `output` 两个子目录，两边文件名一一对应（input 带手写、output 干净），共 2412 对：

```text
<data_root>/
├── 20250211/dataset/{input,output}/     812 对
├── 20250212/dataset/{input,output}/     900 对
└── 20250213/dataset/{input,output}/     700 对
```

格式混杂、尺寸差异、按日期划分的理由见 [`level4_unet/README.md`](../level4_unet/README.md)。

## 为什么不入库

MNIST 一份就有几十 MB，Level 4 的数据集约 1.9 GB，都不适合进版本管理。而且 Level 4 的数据集由招新方分发，仓库里不公开传播下载方式。
