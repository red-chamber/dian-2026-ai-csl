"""配对文档数据集

数据布局
按日期分三批，每批是 input/output 两个目录，文件名一一对应：

    <data_root>/
    ├── 20250211/dataset/{input,output}/
    ├── 20250212/dataset/{input,output}/
    └── 20250213/dataset/{input,output}/

input 是带手写内容的扫描件，output 是同一张图的干净版本（已擦除手写），共 2412 对（812 + 900 + 700）。

特点：

1. input 的格式和通道数不统一：部分文件扩展名是 .jpg 但内容其实是 PNG；另一部分是普通 JPEG 三通道彩色。output 则统一是单通道灰度 JPEG。所以 input 必须统一转灰度
2. 尺寸差异极大：长边从 143 px 到 2000 px，横竖版都有，长宽比各不相同
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 划分用的日期分组
TRAIN_DATES = ("20250211", "20250212")
HOLDOUT_DATE = "20250213"


def scan_pairs(data_root: str | Path, dates: tuple[str, ...] | list[str]) -> list[tuple[Path, Path]]:
    """扫描指定日期下所有 input/output 配对，返回 [(input_path, output_path), ...]
    并保持顺序不变
    """
    data_root = Path(data_root)
    pairs: list[tuple[Path, Path]] = []

    for date in dates:
        input_dir = data_root / date / "dataset" / "input"
        output_dir = data_root / date / "dataset" / "output"
        if not input_dir.is_dir():
            raise FileNotFoundError(
                f"Target dir does not exist: {input_dir}\n"
                f"Expected input: <data_root>/<date>/dataset/{{input,output}}/，"
            )

        # 两边都按文件名建索引，只保留同时存在的
        input_by_stem = {p.stem: p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS}
        output_stems = {p.stem for p in output_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}

        missing = sorted(set(input_by_stem) - output_stems)
        if missing:
            print(f"Warning: In {date} there are {len(missing)} wrong inputs.Skipped.")

        for stem in sorted(input_by_stem):
            if stem in output_stems:
                pairs.append((input_by_stem[stem], output_dir / f"{stem}.jpg"))

    if not pairs:
        raise FileNotFoundError(f"{data_root} 下没有扫到任何配对")

    return pairs


def load_gray(path: Path) -> Image.Image:
    """读图并统一成单通道灰度
    """
    image = Image.open(path)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("L")


class PairedDocDataset(Dataset):
    """配对文档数据集

    Args:
        pairs:      (input_path, output_path) 列表。
        crop_size:  大于 0 时随机裁剪成 crop_size*crop_size（训练用），为 0 时返回整图
        augment:    是否做随机缩放抖动。注意文档图不能水平翻转 —— 文字翻转后不再合理
        min_side:   图像最小边小于 crop_size 时，先等比放大到最小边等于 crop_size，否则裁不出完整窗口
        seed:       裁剪位置的随机源。DataLoader 会给每个 worker 派生不同的种子，但配合 common.utils.set_seed 可以复现
    """

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        *,
        crop_size: int = 0,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.pairs = pairs
        self.crop_size = crop_size
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        input_path, output_path = self.pairs[index]
        input_image = load_gray(input_path)
        target_image = load_gray(output_path)

        # 尺寸不一致时以 input 为准
        if target_image.size != input_image.size:
            target_image = target_image.resize(input_image.size, Image.BILINEAR)

        if self.crop_size:
            input_image, target_image = self._random_crop(input_image, target_image)

        # 转成 [0,1] 的 float32 张量，形状 (1, H, W)
        input_tensor = self._to_tensor(input_image)
        target_tensor = self._to_tensor(target_image)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "name": input_path.stem,
            "size": input_tensor.shape[-2:],
        }

    def _random_crop(self, input_image: Image.Image, target_image: Image.Image):
        """对输入和目标用同一个窗口裁剪"""
        crop = self.crop_size
        width, height = input_image.size

        # 最小边不够就先等比放大
        scale = 1.0
        if self.augment:
            scale *= random.uniform(0.85, 1.15)
        scale = max(scale, crop / min(width, height))
        if scale > 1.0:
            new_size = (max(crop, round(width * scale)), max(crop, round(height * scale)))
            input_image = input_image.resize(new_size, Image.BILINEAR)
            target_image = target_image.resize(new_size, Image.BILINEAR)
            width, height = new_size

        left = random.randint(0, width - crop)
        top = random.randint(0, height - crop)
        box = (left, top, left + crop, top + crop)
        return input_image.crop(box), target_image.crop(box)

    @staticmethod
    def _to_tensor(image: Image.Image):
        array = np.asarray(image, dtype="float32") / 255.0
        return torch.from_numpy(array).unsqueeze(0)  # (H, W) -> (1, H, W)


def split_pairs(
    data_root: str | Path,
    *,
    val_ratio: float = 0.1,
    train_dates=TRAIN_DATES,
    holdout_date: str = HOLDOUT_DATE,
) -> dict[str, list[tuple[Path, Path]]]:
    """按日期划分训练/验证/测试
    返回 {"train": [...], "val": [...], "test": [...]}
    """
    train_pairs = scan_pairs(data_root, train_dates)
    holdout_pairs = scan_pairs(data_root, (holdout_date,))

    n_val = max(1, int(len(holdout_pairs) * val_ratio))
    return {
        "train": train_pairs,
        "val": holdout_pairs[:n_val],
        "test": holdout_pairs[n_val:],
    }


def build_loaders(
    splits: dict[str, list[tuple[Path, Path]]],
    *,
    crop_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
    augment: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, DataLoader]:
    """构建三组 DataLoader

    训练集：固定尺寸的裁剪块，可以批处理
    验证/测试集：整图，尺寸各不相同，所以 batch_size 固定为 1
    """
    loaders: dict[str, DataLoader] = {}

    loaders["train"] = DataLoader(
        PairedDocDataset(splits["train"], crop_size=crop_size, augment=augment, seed=seed),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    for name in ("val", "test"):
        loaders[name] = DataLoader(
            PairedDocDataset(splits[name], crop_size=0, seed=seed),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )

    if verbose:
        for name in ("train", "val", "test"):
            print(f"{name:>5}: {len(splits[name]):,} 对")
        if augment:
            print("数据增强: 随机缩放抖动 0.85~1.15")

    return loaders
