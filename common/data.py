"""数据集注册表与 DataLoader 构建。

三个 Level 的数据管道在这里统一，各 Level 只通过 `dataset` 名字选择数据集：

    mnist           MNIST，28x28 灰度，10 类（Level 1/2 用）
    fashion-mnist   Fashion-MNIST，28x28 灰度，10 类（Level 3 用）

两个数据集的尺寸和通道数完全一样，所以模型可以直接复用；
唯一要跟着换的是归一化用的均值和标准差 —— 这两个常数必须和训练时保持一致，
否则输入分布偏移，预测会不可靠（这也是它们写在注册表里、不做成命令行参数的原因）。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 各数据集的元信息。均值/标准差是各自全集上的统计量（官方公布的常数）。
DATASETS: dict[str, dict] = {
    "mnist": {
        "cls": datasets.MNIST,
        "mean": 0.1307,
        "std": 0.3081,
        "in_channels": 1,
        "input_size": 28,
        "num_classes": 10,
        "label": "MNIST",
        "classes": [str(i) for i in range(10)],
    },
    "fashion-mnist": {
        "cls": datasets.FashionMNIST,
        "mean": 0.2860,
        "std": 0.3530,
        "in_channels": 1,
        "input_size": 28,
        "num_classes": 10,
        "label": "Fashion-MNIST",
        # Fashion-MNIST 的标签是 10 类衣物，不是数字，画图和报错时要用名字
        "classes": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
    },
}


def get_dataset_spec(name: str) -> dict:
    """按名字取数据集元信息，名字写错时列出可选值。"""
    if name not in DATASETS:
        raise ValueError(f"未知数据集 {name!r}，可选：{sorted(DATASETS)}")
    return DATASETS[name]


def build_transforms(spec: dict, augment: bool = False) -> transforms.Compose:
    """构建预处理流水线。

    augment=True 时在训练集上加随机裁剪（先补 2 像素再随机裁回原尺寸）。
    题目目标里提到要理解数据增强，所以这里提供这个开关，但默认关闭：
    开启后训练集的输入分布和 Level 1/2 不再一致，跨 Level 比较会失去可比性。
    """
    steps: list = []
    if augment:
        # 只做随机平移，不做水平翻转 —— 数字和衣服左右翻转后都不再是合理样本
        steps.append(transforms.RandomCrop(spec["input_size"], padding=2))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize((spec["mean"],), (spec["std"],)))
    return transforms.Compose(steps)


def dataset_description(spec: dict, val_split: float) -> str:
    """写进 metrics 的描述，例如 `MNIST (train 90% / val 10% / test 官方 10000)`。"""
    return (
        f"{spec['label']} (train {int((1 - val_split) * 100)}% / "
        f"val {int(val_split * 100)}% / test 官方 10000)"
    )


def build_dataloaders(
    *,
    dataset: str,
    data_dir: str | Path,
    val_split: float,
    batch_size: int,
    seed: int,
    num_workers: int,
    device: str,
    augment: bool = False,
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """构建训练 / 验证 / 测试三个 DataLoader。

    验证集是从训练集里切出来的，val set 和 test set 都不能有随机性。

    返回 (train_loader, val_loader, test_loader, spec)。spec 是数据集元信息，
    调用方要用它建模（in_channels / input_size / num_classes）和写 metrics。

    关于 transform 的处理：
        底层数据集**不带** transform 加载，切分之后再用 Subset + transform
        分别包装训练集和验证集。这样训练集可以单独开数据增强，而验证集始终用
        确定性的预处理 —— 否则每次评估的输入都在随机平移，验证准确率会抖动。
        注意随机切分仍然走 random_split + 固定种子，切出来的划分与早前版本一致。
    """
    spec = get_dataset_spec(dataset)

    eval_transform = build_transforms(spec, augment=False)
    train_transform = build_transforms(spec, augment=augment)

    # data set 路径
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 下载数据集（不带 transform，切分后再分别套）
    base_train = spec["cls"](root=str(data_dir), train=True, download=True)
    test_set = spec["cls"](root=str(data_dir), train=False, download=True, transform=eval_transform)

    # 从训练集里切出验证集
    val_size = int(len(base_train) * val_split)
    train_size = len(base_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_part, val_part = random_split(base_train, [train_size, val_size], generator=generator)

    train_set = _ApplyTransform(train_part, train_transform)
    val_set = _ApplyTransform(val_part, eval_transform)

    # num_workers>0 会用子进程并行读数据；pin_memory 把数据锁在页内存里，传到 GPU 更快
    loader_kwargs = dict(num_workers=num_workers, pin_memory=(device == "cuda"))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **loader_kwargs)

    if verbose:
        print(f"训练集: {len(train_set):,} 张")
        print(f"验证集: {len(val_set):,} 张")
        print(f"测试集: {len(test_set):,} 张")

    return train_loader, val_loader, test_loader, spec


def build_eval_dataset(dataset: str, data_dir: str | Path):
    """加载一份不带任何变换的测试集，用于显示原始图片。

    归一化后的像素是 (x - mean) / std，有正有负，直接用 imshow 显示会变成
    黑白颠倒的噪声图。所以做错误样本分析时要单独加载这一份。
    """
    spec = get_dataset_spec(dataset)
    return spec["cls"](root=str(data_dir), train=False, download=True)


class _ApplyTransform:
    """给不带 transform 的底层数据集套一层固定的 transform。

    数据集返回的是 PIL 图（没给 transform 时），这里补上 ToTensor + Normalize。
    """

    def __init__(self, base_dataset, transform) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, i: int):
        pil_image, label = self.base_dataset[i]
        return self.transform(pil_image), label
