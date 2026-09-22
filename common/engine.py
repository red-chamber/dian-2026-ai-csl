"""训练循环、评估与完整的训练流程。

三个 Level 的 train.py 都调用这里的 `run_training`，所以「每轮训练 + 验证、
按验证准确率保存最优权重、训练结束后在测试集上评估一次」这套流程只写一遍。
各 Level 的 train.py 剩下的是命令行接口、建模型、以及自己特有的输出。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .utils import gpu_peak_memory_mb


class TrainResult(NamedTuple):
    """一次训练的产出，字段名和 reports/metrics/*.json 里的键对应。"""

    history: dict
    best_epoch: int
    best_val_acc: float
    test_loss: float
    test_acc: float
    train_time_sec: float
    gpu_peak_mb: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """跑一个训练 epoch，返回 (平均损失, 准确率)

        1. optimizer.zero_grad()  清空上一轮累积的梯度
        2. loss.backward()        反向传播：从损失出发，用链式法则算出每个参数的梯度
        3. optimizer.step()       按梯度方向更新参数
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        # 数据搬到 GPU（如果可用）
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # --- 前向计算 ---
        logits = model(images)
        # softmax + 取负对数似然
        loss = criterion(logits, labels)

        # --- 反向传播与参数更新 ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()  # 评估不需要反向传播，所以关掉自动求导的记录，省显存
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str
) -> tuple[float, float]:
    """评估，返回 (平均损失, 准确率)

    """
    model.eval()  # Dropout 关闭

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def build_optimizer(
    model: nn.Module,
    *,
    optimizer: str,
    lr: float,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """按名字建优化器。"""
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(f"未知优化器 {optimizer!r}，可选：['adam', 'sgd']")


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    device: str,
    ckpt_path: str | Path,
    model_config: dict,
    args_dict: dict,
    verbose: bool = True,
) -> TrainResult:
    """完整跑一遍训练流程。

    步骤：
        1. 逐轮训练 + 验证，记录 history
        2. 只在验证准确率**创下新高**时保存权重 —— 训练准确率还在涨、验证反而
           下降就是过拟合了，存最优而不是存最后一轮是基本操作
        3. 训练全部结束后，加载最优权重在测试集上评估**一次**
           （测试集不参与任何决策，这样才不算数据泄露）
        4. 把 test_acc 补写回 checkpoint
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history: dict[str, list] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = 0

    if verbose:
        print("\n" + "-" * 60)
        print(f"{'Epoch':>7} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>9} {'Time':>7}")
        print("-" * 60)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 只保存验证集上最好的模型
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                ckpt_path,
                model=model,
                model_config=model_config,
                epoch=epoch,
                val_acc=val_acc,
                args=args_dict,
            )
            marker = "  <- best"

        if verbose:
            print(
                f"{epoch:>7} {train_loss:>11.4f} {train_acc:>9.2%} "
                f"{val_loss:>9.4f} {val_acc:>9.2%} {dt:>6.1f}s{marker}"
            )

    total_time = time.time() - start_time
    if verbose:
        print("-" * 60)
        print(
            f"训练完成，耗时 {total_time:.1f}s，"
            f"最好的 epoch 是第 {best_epoch} 轮（验证准确率 {best_val_acc:.2%}）"
        )

    # ---------- 加载最好权重，在测试集上评估 ----------
    if verbose:
        print("\n在测试集上评估最好权重...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    if verbose:
        print(f"测试集准确率：{test_acc:.2%}   测试集损失：{test_loss:.4f}")

    # 把测试准确率补回 checkpoint
    ckpt["test_acc"] = test_acc
    torch.save(ckpt, ckpt_path)

    return TrainResult(
        history=history,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        test_loss=test_loss,
        test_acc=test_acc,
        train_time_sec=total_time,
        gpu_peak_mb=gpu_peak_memory_mb(device),
    )
