"""训练 + 验证脚本（Level 3：AlexNet / ResNet on Fashion-MNIST）

数据加载、训练循环、画曲线、指标导出全部复用 common/ 里的公共实现，
本文件只负责三件事：命令行接口、按参数建模型、训练后打印实验摘要。

验收标准：
1. 在已有代码基础上分别实现 AlexNet、ResNet
2. README 记录网络结构与超参数

outputs:
    checkpoints/<tag>_best.pt              验证集上最好的权重
    reports/figures/<tag>_curves.png       Loss / Accuracy 曲线
    reports/metrics/<tag>.json             本次实验的全部指标

用法：
    # 默认数据集是 Fashion-MNIST，比 MNIST 难，深层网络的优势才看得出来
    python level3_classic_networks/src/train.py --model alexnet
    python level3_classic_networks/src/train.py --model resnet

    # 消融：去掉残差连接，看深了会怎样
    python level3_classic_networks/src/train.py --model resnet --no-residual --tag resnet_plain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# 本文件是 `python level3_classic_networks/src/train.py` 这样直接运行的，
# Python 只会把脚本所在目录放进 sys.path，仓库根不在里面，所以 common/ 找不到。
# 这几行把仓库根补进去。
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
    """按命令行参数建模型，规格（输入通道/尺寸/类别数）由数据集决定。"""
    if args.model == "alexnet":
        # 原版 AlexNet 在全连接层用 0.5 的 Dropout，这里作为默认值
        dropout = 0.5 if args.dropout is None else args.dropout
        return ARCHITECTURES["alexnet"](
            in_channels=spec["in_channels"],
            input_size=spec["input_size"],
            num_classes=spec["num_classes"],
            dropout=dropout,
        )

    # ResNet 靠 BatchNorm 做正则，原版不加 Dropout
    return ARCHITECTURES["resnet"](
        in_channels=spec["in_channels"],
        input_size=spec["input_size"],
        num_classes=spec["num_classes"],
        residual=not args.no_residual,
    )


def model_display_name(args: argparse.Namespace) -> str:
    """写进 metrics 的模型名，能把消融变体区分开。"""
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
    print(f"设备     : {args.device}")
    if args.device == "cuda":
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"随机种子 : {args.seed}")

    # ---------- 数据 ----------
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
    if args.augment:
        print("数据增强 : 训练集随机裁剪 padding=2")

    # ---------- 模型 ----------
    model = build_model(args, spec).to(args.device)
    n_params = count_parameters(model)
    print(f"\n模型     : {model_name}")
    print(f"参数量   : {n_params:,}")

    # ---------- 损失函数与优化器 ----------
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        optimizer=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    print(f"优化器   : {args.optimizer.upper()}，lr={args.lr}，weight_decay={args.weight_decay}")

    # ---------- 训练 ----------
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

    # ---------- 画曲线 ----------
    fig_path = PROJECT_ROOT / "reports" / "figures" / f"{tag}_curves.png"
    plot_curves(result.history, fig_path, f"{model_name} on {spec['label']}")

    # ---------- 保存本次实验的全部指标 ----------
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
    print(f"实验指标已保存：{metrics_path}")

    # ---------- 打印实验记录摘要 ----------
    print("\n" + "=" * 60)
    print("实验记录（可直接填进 docs/experiment-log.md）")
    print("=" * 60)
    print(f"实验编号与日期  : {tag} / {metrics['date']}")
    print(f"Git commit      : {metrics['git_commit']}")
    print(f"模型名称        : {model_name}（{n_params:,} 参数）")
    print(f"数据集与划分    : {metrics['dataset']}")
    print(f"随机种子        : {args.seed}")
    print(f"输入尺寸        : 1x28x28")
    print(f"batch size      : {args.batch_size}")
    print(f"优化器/学习率   : {args.optimizer} / {args.lr}")
    print(f"损失函数        : CrossEntropyLoss")
    print(f"训练轮数        : {args.epochs}（最优在第 {result.best_epoch} 轮）")
    print(f"验证集准确率    : {result.best_val_acc:.2%}")
    print(f"测试集准确率    : {result.test_acc:.2%}")
    print(f"训练时长        : {result.train_time_sec:.1f}s")
    print(f"GPU 显存峰值    : {result.gpu_peak_mb:.1f} MB")
    print(f"结果图路径      : {metrics['figure']}")
    print("=" * 60)
    print("\n下一步：另一个模型也训练一遍，然后运行 compare.py 做对比")


if __name__ == "__main__":
    main()
