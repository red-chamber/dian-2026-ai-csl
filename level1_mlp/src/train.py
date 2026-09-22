"""训练 + 验证脚本（Level 1：MLP on MNIST）

数据加载、训练循环、画曲线、指标导出都复用 common/ 的公共实现，
本文件只保留三件事：命令行接口、建模型、训练后打印实验摘要。

验收标准：
1. 准确率达到 90%
2. 能训练模型、推理模型，并可视化、保存 Loss 曲线
3. 编写 README，记录网络结构、超参数和实验结果

outputs:
    checkpoints/<tag>_best.pt              验证集上最好的权重
    reports/figures/<tag>_curves.png       Loss / Accuracy 曲线
    reports/metrics/<tag>.json             本次实验的全部指标，用于填实验记录

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# 脚本是 `python level1_mlp/src/train.py` 这样直接运行的，Python 只会把脚本所在
# 目录放进 sys.path，仓库根不在里面，所以 common/ 找不到。这几行补上。
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.data import build_dataloaders, dataset_description  # noqa: E402
from common.engine import build_optimizer, run_training  # noqa: E402
from common.plots import plot_curves  # noqa: E402
from common.utils import PROJECT_ROOT, build_metrics, count_parameters, save_metrics, set_seed  # noqa: E402
from model import MLP  # noqa: E402


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
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion-mnist"],
                   help="默认 MNIST；换成 fashion-mnist 可以看同一个模型的难度上限")
    p.add_argument("--val-split", type=float, default=0.1, help="从训练集里切多少比例作验证集")
    p.add_argument("--seed", type=int, default=42, help="随机种子，固定后结果可复现")
    p.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "raw"))
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader 的进程数")
    # --- 其他 ---
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", type=str, default="mlp_mnist", help="产物文件名前缀")
    return p.parse_args()


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
    train_loader, val_loader, test_loader, spec = build_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        val_split=args.val_split,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )

    # ---------- 模型 ----------
    model = MLP(
        hidden_sizes=tuple(args.hidden_sizes),
        in_features=spec["input_size"] * spec["input_size"],
        num_classes=spec["num_classes"],
        dropout=args.dropout,
    ).to(args.device)

    n_params = count_parameters(model)
    print(f"\n模型结构 : MLP {args.hidden_sizes}，Dropout={args.dropout}")
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
    ckpt_path = PROJECT_ROOT / "checkpoints" / f"{args.tag}_best.pt"
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
    fig_path = PROJECT_ROOT / "reports" / "figures" / f"{args.tag}_curves.png"
    plot_curves(result.history, fig_path, f"MLP {args.hidden_sizes} on {spec['label']}",
                target=0.90, target_label="90% target")

    # ---------- 保存本次实验的全部指标 ----------
    metrics = build_metrics(
        args,
        model_name=f"MLP {args.hidden_sizes}",
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
        extra={"dropout": args.dropout, "hidden_sizes": list(args.hidden_sizes)},
    )
    metrics_path = save_metrics(metrics, args.tag)
    print(f"实验指标已保存：{metrics_path}")

    # ---------- 打印实验记录摘要 ----------
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
    print(f"训练轮数        : {args.epochs}（最优在第 {result.best_epoch} 轮）")
    print(f"验证集准确率    : {result.best_val_acc:.2%}")
    print(f"测试集准确率    : {result.test_acc:.2%}   "
          f"{'✓ 达到 90% 的验收标准' if result.test_acc >= 0.90 else '✗ 未达到 90%'}")
    print(f"训练时长        : {result.train_time_sec:.1f}s")
    print(f"GPU 显存峰值    : {result.gpu_peak_mb:.1f} MB")
    print(f"结果图路径      : {metrics['figure']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
