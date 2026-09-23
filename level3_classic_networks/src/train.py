"""训练 + 验证脚本（Level 3：AlexNet / ResNet on Fashion-MNIST）

本文件：命令行接口、按参数建模型
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.data import build_dataloaders, dataset_description  # noqa: E402
from common.engine import build_optimizer, run_training  # noqa: E402
from common.plots import plot_curves  # noqa: E402
from common.utils import PROJECT_ROOT, build_metrics, count_parameters, save_metrics, set_seed  # noqa: E402
from model import ARCHITECTURES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练 AlexNet 或 ResNet 完成图像分类")
    # --- 模型 ---
    p.add_argument("--model", type=str, default="resnet", choices=sorted(ARCHITECTURES),
                   help="用哪个经典网络")
    p.add_argument("--no-residual", action="store_true",
                   help="ResNet 专用：去掉跨层连接，得到一个同深度的 PlainNet（消融用）")
    # --- 训练相关 ---
    p.add_argument("--epochs", type=int, default=20, help="训练轮数（深层网络比 MLP 需要更多轮）")
    p.add_argument("--batch-size", type=int, default=128, help="每个 batch 的图片数")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--momentum", type=float, default=0.9, help="SGD 的动量（仅 --optimizer sgd 时生效）")
    p.add_argument("--weight-decay", type=float, default=0.0, help="L2 正则系数")
    p.add_argument("--dropout", type=float, default=None,
                   help="Dropout 概率；不传时 AlexNet 用 0.5（原版设定），ResNet 不用 Dropout")
    # --- 数据相关 ---
    p.add_argument("--dataset", type=str, default="fashion-mnist", choices=["mnist", "fashion-mnist"])
    p.add_argument("--augment", action="store_true",
                   help="训练集加随机裁剪（默认关闭，保持与 Level 1/2 输入分布一致）")
    p.add_argument("--val-split", type=float, default=0.1, help="从训练集里切多少比例作验证集")
    p.add_argument("--seed", type=int, default=42, help="随机种子，固定后结果可复现")
    p.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "raw"))
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader 的进程数")
    # --- 其他 ---
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", type=str, default=None, help="产物文件名前缀，默认按模型和数据集自动生成")
    return p.parse_args()


def build_model(args: argparse.Namespace, spec: dict) -> nn.Module:
    """按命令行参数建模型
    """
    if args.model == "alexnet":
        dropout = 0.5 if args.dropout is None else args.dropout
        return ARCHITECTURES["alexnet"](
            in_channels=spec["in_channels"],
            input_size=spec["input_size"],
            num_classes=spec["num_classes"],
            dropout=dropout,
        )

    return ARCHITECTURES["resnet"](
        in_channels=spec["in_channels"],
        input_size=spec["input_size"],
        num_classes=spec["num_classes"],
        residual=not args.no_residual,
    )


def model_display_name(args: argparse.Namespace) -> str:
    """写进 metrics 的模型名"""
    if args.model == "alexnet":
        return f"AlexNet (dropout={0.5 if args.dropout is None else args.dropout})"
    return "ResNet18" if not args.no_residual else "PlainNet18 (no residual)"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tag = args.tag or f"{args.model}_{'fashionmnist' if args.dataset == 'fashion-mnist' else 'mnist'}"
    model_name = model_display_name(args)

    print("=" * 60)
    print("Level 3：经典网络 AlexNet / ResNet")
    print("=" * 60)
    print(f"Device: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Random seed: {args.seed}")

    train_loader, val_loader, test_loader, spec = build_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        val_split=args.val_split,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        augment=args.augment,
    )

    model = build_model(args, spec).to(args.device)
    n_params = count_parameters(model)
    print(f"\nModel name: {model_name}")
    print(f"Parameter count: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        optimizer=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    print(f"优化器   : {args.optimizer.upper()}，lr={args.lr}，weight_decay={args.weight_decay}")

    ckpt_path = PROJECT_ROOT / "checkpoints" / f"{tag}_best.pt"
    result = run_training(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        epochs=args.epochs,
        device=args.device,
        ckpt_path=ckpt_path,
        model_config=model.config(),
        args_dict=vars(args),
    )

    # Plotting
    fig_path = PROJECT_ROOT / "reports" / "figures" / f"{tag}_curves.png"
    plot_curves(result.history, fig_path, f"{model_name} on {spec['label']}")

    # Save to metrics
    metrics = build_metrics(
        args,
        model_name=model_name,
        n_params=n_params,
        dataset_desc=dataset_description(spec, args.val_split),
        input_size=[spec["in_channels"], spec["input_size"], spec["input_size"]],
        history=result.history,
        best_epoch=result.best_epoch,
        best_val_acc=result.best_val_acc,
        test_loss=result.test_loss,
        test_acc=result.test_acc,
        train_time_sec=result.train_time_sec,
        gpu_peak_mb=result.gpu_peak_mb,
        figure=fig_path,
        checkpoint=ckpt_path,
        extra={
            "arch": args.model,
            "residual": not args.no_residual if args.model == "resnet" else None,
            "augment": args.augment,
        },
    )
    metrics_path = save_metrics(metrics, tag)
    print(f"Saved in {metrics_path}")

if __name__ == "__main__":
    main()
