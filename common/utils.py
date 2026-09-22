"""路径常量、随机种子、参数量统计、指标组装与导出。"""

from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# common/utils.py 在 <仓库根>/common/ 下，往上一级就是仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """固定所有随机源，保证同一个种子跑出来的结果一致。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """统计参数量。

    三个 Level 共用同一个口径（numel() 累加），这样 MLP / CNN / AlexNet /
    ResNet 的参数量可以直接放在一张表里比。
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_git_commit() -> str:
    """取当前 commit id，方便把实验结果和代码版本对应起来。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def env_info() -> dict:
    """记录运行环境，便于日后复现。"""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


def gpu_peak_memory_mb(device: str) -> float:
    """GPU 显存峰值（MB）；用 CPU 时返回 0。"""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def build_metrics(
    args,
    *,
    model_name: str,
    n_params: int,
    dataset_desc: str,
    input_size: list[int],
    history: dict,
    best_epoch: int,
    best_val_acc: float,
    test_loss: float,
    test_acc: float,
    train_time_sec: float,
    gpu_peak_mb: float,
    figure: Path,
    checkpoint: Path,
    extra: dict | None = None,
) -> dict:
    """组装一份实验指标，字段与 docs/experiment-log.md 的清单一致。

    三个 Level 用同一个 schema，json 可以直接横向比较。
    各 Level 特有的超参（比如 Level 1/2 的 dropout）通过 extra 传进来。
    """
    metrics = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(),
        "model": model_name,
        "params": n_params,
        "dataset": dataset_desc,
        "seed": args.seed,
        "input_size": list(input_size),
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "loss_fn": "CrossEntropyLoss",
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train_time_sec": round(train_time_sec, 2),
        "gpu_peak_mem_mb": round(gpu_peak_mb, 1),
        "history": history,
        "figure": str(figure.relative_to(PROJECT_ROOT)),
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "env": env_info(),
    }
    if extra:
        metrics.update(extra)
    return metrics


def save_metrics(metrics: dict, tag: str) -> Path:
    """把指标写成 reports/metrics/<tag>.json。"""
    path = PROJECT_ROOT / "reports" / "metrics" / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


def load_metrics(tag: str) -> dict:
    """读取训练时导出的指标 json（用于取逐轮 history）。"""
    path = PROJECT_ROOT / "reports" / "metrics" / f"{tag}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到实验指标：{path}\n请先训练对应模型（train.py）再运行本脚本。"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)
