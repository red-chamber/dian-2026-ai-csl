"""Level 1 的训练 + 验证脚本：用 MLP 完成 MNIST 手写数字识别。

对应题目的验收标准：
  - 在不数据泄露的情况下，测试集准确率达到 90%
  - 能训练模型、推理模型，并可视化、保存 Loss 曲线
  - README 中记录网络结构、超参数和实验结果

用法：
    conda activate dian-ai
    python level1_mlp/src/train.py                    # 用默认超参数跑
    python level1_mlp/src/train.py --epochs 15 --lr 5e-4 --dropout 0.3

产物：
    checkpoints/mlp_mnist_best.pt              验证集上最好的权重
    reports/figures/mlp_mnist_curves.png       Loss 曲线（训练 vs 验证）+ 准确率曲线
    reports/metrics/mlp_mnist.json             本次实验的全部指标，用于填实验记录


=================== 关于「不数据泄露」===================

题目特意强调了这一条，因为它是新手最容易犯、又最不容易发现的错误。三种典型泄露：

  1. 用测试集调超参数。如果看着测试集准确率去调学习率、层数，那测试集就变成了
     验证集，报出来的数字不再是「没见过的数据上的表现」，会偏乐观。
     → 本脚本的做法：从**训练集**里切出 10% 作验证集，测试集全程封存，
       只在最后加载最好权重时评估一次。

  2. 在划分数据集之前做归一化。用全量数据的均值方差去归一化，等于让模型
     间接「看到」了测试集的统计信息。
     → 本脚本的做法：直接用 MNIST 官方公布的均值 0.1307 / 标准差 0.3081，
       这是公开常数，不依赖任何本地数据，从根上避免这个问题。

  3. 训练时用了随机增强，验证/测试时也用了。验证和测试必须走 deterministic 的
     预处理，否则同一个模型每次评估结果都在抖。
     → 本脚本的做法：三个数据集用同一套 ToTensor + Normalize，不含任何随机变换。

"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无图形界面环境（如纯终端、服务器）也能出图

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import MLP, count_parameters

# 脚本在 level1_mlp/src/ 下，往上两级是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MNIST 全集的均值和标准差（官方公布的常数）。
# 归一化的目的：把输入缩放到均值 0、方差 1 附近，让各层输入分布稳定，训练更稳更快。
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练 MLP 完成 MNIST 分类")
    # --- 训练相关 ---
    p.add_argument("--epochs", type=int, default=10, help="训练轮数")
    p.add_argument("--batch-size", type=int, default=128, help="每个 batch 的图片数")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--momentum", type=float, default=0.9, help="SGD 的动量（仅 --optimizer sgd 时生效）")
    p.add_argument("--weight-decay", type=float, default=0.0, help="L2 正则系数")
    # --- 模型结构 ---
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 256], help="各隐藏层宽度")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout 概率，0 表示不用")
    # --- 数据相关 ---
    p.add_argument("--val-split", type=float, default=0.1, help="从训练集里切多少比例作验证集")
    p.add_argument("--seed", type=int, default=42, help="随机种子，固定后结果可复现")
    p.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "raw"))
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader 的进程数")
    # --- 调试用 ---
    p.add_argument("--limit-train", type=int, default=None, help="只用前 N 张训练图（调试用）")
    p.add_argument("--limit-test", type=int, default=None, help="只用前 N 张测试图（调试用）")
    # --- 其他 ---
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", type=str, default="mlp_mnist", help="产物文件名前缀")
    return p.parse_args()


def set_seed(seed: int) -> None:
    """固定所有随机源，保证同一个种子跑出来的结果一致。

    需要同时固定这么多地方，是因为随机性来自多个独立来源：
    Python 的 random、NumPy、PyTorch 的 CPU 和 GPU 随机数、以及 cuDNN 的算法选择。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 让 cuDNN 只用确定性算法。会略微变慢，但换来可复现。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练 / 验证 / 测试三个 DataLoader。

    关键点：验证集是从**训练集**里切出来的，测试集完全不参与训练过程。
    """
    # 训练和验证/测试用同一套预处理：转张量 + 归一化。
    # 之所以不在这里加随机增强，是因为验证和测试不能有随机性。
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # download=True 会在首次运行时自动下载（约 12 MB）
    full_train = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)

    # 从训练集里切出验证集。用带种子的 generator，保证每次划分结果相同。
    val_size = int(len(full_train) * args.val_split)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    # 调试模式：只取一小部分，快速验证流程能不能跑通
    if args.limit_train is not None:
        train_set = torch.utils.data.Subset(train_set, range(min(args.limit_train, len(train_set))))
    if args.limit_test is not None:
        test_set = torch.utils.data.Subset(test_set, range(min(args.limit_test, len(test_set))))

    # num_workers>0 会用子进程并行读数据；pin_memory 把数据锁在页内存里，传到 GPU 更快
    common = dict(num_workers=args.num_workers, pin_memory=(args.device == "cuda"))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **common)

    print(f"训练集: {len(train_set):,} 张")
    print(f"验证集: {len(val_set):,} 张  （从训练集切出 {args.val_split:.0%}）")
    print(f"测试集: {len(test_set):,} 张  （全程封存，最后才用）")

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """跑一个训练 epoch，返回（平均损失, 准确率）。

    这就是「神经网络怎么通过训练让结果越来越正确」的完整答案，只有四步：

        1. optimizer.zero_grad()  清空上一轮累积的梯度
        2. loss.backward()        反向传播：从损失出发，用链式法则算出每个参数的梯度
        3. optimizer.step()       按梯度方向更新参数
        4. 重复                   一轮轮下来，损失下降、准确率上升

    第 2 步是整件事的核心：PyTorch 的自动求导会记录前向计算经过的每一步，
    反向走一遍就能得到「每个参数该往哪个方向调、调多少」。
    """
    model.train()  # 训练模式：Dropout 生效

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        # 数据搬到 GPU（如果可用）。这一步是最常见的性能瓶颈来源。
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # --- 前向计算：图片 -> 10 个类别打分 ---
        logits = model(images)
        # CrossEntropyLoss 内部 = softmax + 取负对数似然，所以这里直接喂原始打分
        loss = criterion(logits, labels)

        # --- 反向传播与参数更新 ---
        optimizer.zero_grad()  # 梯度是累加的，每轮必须清零
        loss.backward()        # 求梯度
        optimizer.step()       # 更新参数

        # --- 统计 ---
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str
) -> tuple[float, float]:
    """评估，返回（平均损失, 准确率）。

    @torch.no_grad() 关掉自动求导的记录——评估不需要反向传播，
    关掉能省显存、跑得更快。
    """
    model.eval()  # 评估模式：Dropout 关闭，行为确定

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


def plot_curves(history: dict, out_path: Path, title: str) -> None:
    """画 Loss 曲线和准确率曲线并保存。

    题目要求「可视化、保存 Loss 曲线」，这是其中最重要的一张图：
    训练损失一直降但验证损失开始涨，就是过拟合的信号。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图：Loss 曲线
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train", color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Validation", color="#d62728")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 右图：准确率曲线
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train", color="#1f77b4")
    axes[1].plot(epochs, history["val_acc"], "s-", label="Validation", color="#d62728")
    axes[1].axhline(0.90, ls="--", lw=1, color="gray", label="90% target")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"曲线已保存：{out_path}")


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("Level 1：MLP 完成 MNIST 手写数字识别")
    print("=" * 60)
    print(f"设备     : {args.device}")
    if args.device == "cuda":
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"随机种子 : {args.seed}")

    # ---------- 数据 ----------
    train_loader, val_loader, test_loader = build_dataloaders(args)

    # ---------- 模型 ----------
    model = MLP(
        hidden_sizes=tuple(args.hidden_sizes),
        in_features=28 * 28,
        num_classes=10,
        dropout=args.dropout,
    ).to(args.device)

    n_params = count_parameters(model)
    print(f"\n模型结构 : MLP {args.hidden_sizes}，Dropout={args.dropout}")
    print(f"参数量   : {n_params:,}")

    # ---------- 损失函数与优化器 ----------
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    print(f"优化器   : {args.optimizer.upper()}，lr={args.lr}，weight_decay={args.weight_decay}")

    # ---------- 训练循环 ----------
    ckpt_path = PROJECT_ROOT / "checkpoints" / f"{args.tag}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history: dict[str, list] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = 0

    print("\n" + "-" * 60)
    print(f"{'Epoch':>7} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>9} {'Time':>7}")
    print("-" * 60)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 只保存验证集上最好的模型。
        # 不要保存最后一个 epoch 的——训练后期往往已经过拟合，验证集表现反而会变差。
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.config(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                ckpt_path,
            )
            marker = "  <- best"

        print(
            f"{epoch:>7} {train_loss:>11.4f} {train_acc:>9.2%} "
            f"{val_loss:>9.4f} {val_acc:>9.2%} {dt:>6.1f}s{marker}"
        )

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"训练完成，耗时 {total_time:.1f}s，最好的 epoch 是第 {best_epoch} 轮（验证准确率 {best_val_acc:.2%}）")

    # ---------- 加载最好权重，在测试集上评估一次 ----------
    # 这一步之前，测试集一次都没被碰过。
    print("\n在测试集上评估最好权重（测试集只在这一次使用）...")
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)
    print(f"测试集准确率：{test_acc:.2%}   测试集损失：{test_loss:.4f}")

    # 把测试准确率补回 checkpoint，推理脚本要用
    ckpt["test_acc"] = test_acc
    torch.save(ckpt, ckpt_path)

    # ---------- 画曲线 ----------
    fig_path = PROJECT_ROOT / "reports" / "figures" / f"{args.tag}_curves.png"
    plot_curves(history, fig_path, f"MLP {args.hidden_sizes} on MNIST")

    # ---------- 保存本次实验的全部指标 ----------
    gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else 0.0
    metrics = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(),
        "model": f"MLP {args.hidden_sizes}",
        "params": n_params,
        "dataset": f"MNIST (train {int((1 - args.val_split) * 100)}% / val {int(args.val_split * 100)}% / test 官方 10000)",
        "seed": args.seed,
        "input_size": [1, 28, 28],
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "loss_fn": "CrossEntropyLoss",
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train_time_sec": round(total_time, 2),
        "gpu_peak_mem_mb": round(gpu_peak_mb, 1),
        "history": history,
        "figure": str(fig_path.relative_to(PROJECT_ROOT)),
        "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
        },
    }

    metrics_path = PROJECT_ROOT / "reports" / "metrics" / f"{args.tag}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"实验指标已保存：{metrics_path}")

    # ---------- 打印实验记录摘要，可直接贴进 docs/experiment-log.md ----------
    print("\n" + "=" * 60)
    print("实验记录（可直接填进 docs/experiment-log.md）")
    print("=" * 60)
    print(f"实验编号与日期  : {args.tag} / {metrics['date']}")
    print(f"Git commit      : {metrics['git_commit']}")
    print(f"模型名称        : {metrics['model']}（{n_params:,} 参数）")
    print(f"数据集与划分    : {metrics['dataset']}")
    print(f"随机种子        : {args.seed}")
    print(f"输入尺寸        : 1x28x28（展平为 784）")
    print(f"batch size      : {args.batch_size}")
    print(f"优化器/学习率   : {args.optimizer} / {args.lr}")
    print(f"损失函数        : CrossEntropyLoss")
    print(f"训练轮数        : {args.epochs}（最优在第 {best_epoch} 轮）")
    print(f"验证集准确率    : {best_val_acc:.2%}")
    print(f"测试集准确率    : {test_acc:.2%}   {'✓ 达到 90% 的验收标准' if test_acc >= 0.90 else '✗ 未达到 90%'}")
    print(f"训练时长        : {total_time:.1f}s")
    print(f"GPU 显存峰值    : {gpu_peak_mb:.1f} MB")
    print(f"结果图路径      : {metrics['figure']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
