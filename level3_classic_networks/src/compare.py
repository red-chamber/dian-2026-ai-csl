"""经典网络对比脚本（Level 3）

把训练过的若干模型放在一起比，回答这一 Level 的核心问题：
「AlexNet 和 ResNet 的思路差在哪，各自换来什么」。

本脚本**不重新训练**，只读各模型训练时导出的 reports/metrics/<tag>.json，
所以跑一次只要几秒。

    python level3_classic_networks/src/compare.py
    python level3_classic_networks/src/compare.py --tags alexnet_fashionmnist resnet_fashionmnist resnet_plain

对比的角度：
    参数量      用同一个 count_parameters 口径，横向可比
    测试准确率  各自在官方测试集 10000 张上评估一次得到（训练脚本里记录的）
    收敛速度    读逐轮 history，看第几轮达到某个验证准确率
    参数量构成  卷积部分 vs 分类头各占多少 —— 这一项最能说明两个网络的设计差异

outputs:
    reports/figures/level3_params_vs_acc.png     参数量 / 准确率对比条形图
    reports/figures/level3_convergence.png       各模型的验证准确率曲线叠加
    reports/metrics/compare_level3.json          全部对比数字
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.plots import COMPARE_COLORS, plot_compare, save_figure  # noqa: E402
from common.utils import PROJECT_ROOT, load_metrics  # noqa: E402

DEFAULT_TAGS = ["alexnet_fashionmnist", "resnet_fashionmnist"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="对比若干经典网络在同一个数据集上的表现")
    p.add_argument("--tags", type=str, nargs="+", default=DEFAULT_TAGS,
                   help="要对比的 metrics 标签（对应 reports/metrics/<tag>.json）")
    p.add_argument("--target", type=float, default=0.92,
                   help="画一条水平参考线，标出希望达到的验证准确率")
    p.add_argument("--top-k", type=int, default=3, help="每个模型打印前 K 个易混类别对")
    return p.parse_args()


def epochs_to_reach(history: dict, threshold: float) -> int | None:
    """第一次达到某个验证准确率是在第几轮；达不到返回 None。"""
    for i, acc in enumerate(history["val_acc"], start=1):
        if acc >= threshold:
            return i
    return None


def load_all(tags: list[str]) -> dict[str, dict]:
    """逐个读 metrics；缺哪个就跳过并提示，不因为一个缺失就整体失败。"""
    loaded: dict[str, dict] = {}
    for tag in tags:
        try:
            loaded[tag] = load_metrics(tag)
        except FileNotFoundError:
            print(f"跳过 {tag}：还没训练过（找不到 reports/metrics/{tag}.json）")
    if not loaded:
        raise SystemExit(
            "没有任何可对比的模型。请先训练：\n"
            "  python level3_classic_networks/src/train.py --model alexnet\n"
            "  python level3_classic_networks/src/train.py --model resnet"
        )
    return loaded


def plot_params_vs_accuracy(metrics: dict[str, dict], out_path: Path) -> None:
    """左图参数量、右图测试准确率，两个模型的柱子一一对应。"""
    names = [m["model"] for m in metrics.values()]
    params = [m["params"] for m in metrics.values()]
    accs = [m["test_acc"] for m in metrics.values()]
    colors = [COMPARE_COLORS[i % len(COMPARE_COLORS)] for i in range(len(names))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    bars = axes[0].bar(range(len(names)), params, color=colors)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylabel("Parameters")
    axes[0].set_title("Parameter count")
    axes[0].grid(alpha=0.3, axis="y")
    for bar, value in zip(bars, params):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}",
                     ha="center", va="bottom", fontsize=8)

    bars = axes[1].bar(range(len(names)), accs, color=colors)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_ylim(min(accs) - 0.05 if min(accs) > 0.05 else 0, 1.0)
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Test accuracy")
    axes[1].grid(alpha=0.3, axis="y")
    for bar, value in zip(bars, accs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2%}",
                     ha="center", va="bottom", fontsize=9)

    fig.suptitle("Classic networks: parameters vs accuracy", fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_path)


def main() -> None:
    args = parse_args()
    metrics = load_all(args.tags)

    print("=" * 70)
    print("Level 3：经典网络对比")
    print("=" * 70)

    # ---------- 参数量与准确率 ----------
    print(f"\n{'模型':<26}{'参数量':>12}{'测试准确率':>12}{'最优轮次':>10}{'每轮耗时':>10}")
    print("-" * 70)
    for tag, m in metrics.items():
        sec_per_epoch = m["train_time_sec"] / m["epochs"]
        print(
            f"{m['model']:<26}{m['params']:>12,}{m['test_acc']:>12.2%}"
            f"{m['best_epoch']:>10}{sec_per_epoch:>9.1f}s"
        )

    # ---------- 收敛速度 ----------
    print(f"\n验证准确率第一次达到 {args.target:.0%} 是在第几轮：")
    reach = {}
    for tag, m in metrics.items():
        epoch = epochs_to_reach(m["history"], args.target)
        reach[tag] = epoch
        print(f"  {m['model']:<26}{f'第 {epoch} 轮' if epoch else '未达到'}")
        print(f"    第 1 轮 {m['history']['val_acc'][0]:.2%} "
              f"-> 最优第 {m['best_epoch']} 轮 {m['best_val_acc']:.2%}")

    # ---------- 画图 ----------
    fig_dir = PROJECT_ROOT / "reports" / "figures"

    plot_params_vs_accuracy(metrics, fig_dir / "level3_params_vs_acc.png")

    plot_compare(
        {m["model"]: m["history"] for m in metrics.values()},
        fig_dir / "level3_convergence.png",
        f"Convergence on {list(metrics.values())[0]['dataset'].split(' ')[0]}",
        metric="val_acc",
        ylabel="Validation accuracy",
        target=args.target,
        target_label=f"{args.target:.0%} target",
    )

    # ---------- 存 JSON ----------
    result = {
        tag: {
            "model": m["model"],
            "params": m["params"],
            "test_acc": m["test_acc"],
            "test_loss": m["test_loss"],
            "best_epoch": m["best_epoch"],
            "best_val_acc": m["best_val_acc"],
            "train_time_sec": m["train_time_sec"],
            "sec_per_epoch": round(m["train_time_sec"] / m["epochs"], 2),
            "epoch_to_target": reach[tag],
        }
        for tag, m in metrics.items()
    }
    out_path = PROJECT_ROOT / "reports" / "metrics" / "compare_level3.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存：{out_path}")


if __name__ == "__main__":
    main()
