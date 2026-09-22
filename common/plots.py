"""通用绘图。

matplotlib 的坑集中在文件开头这几行，所以统一在这里处理一次：

    1. matplotlib.use("Agg") 必须在 import pyplot **之前** —— pyplot 一被导入，
       backend 就锁定了。Agg 表示「只写文件、不弹窗」，WSL / SSH 等没有显示器的
       环境下必须用它，否则弹窗失败或直接卡住。
    2. savefig 不会自动创建目录，所以每次存图前都要 mkdir(parents=True)。
    3. 用完的 figure 要 plt.close() 释放 —— 循环里画图不关会持续累积内存。
    4. 默认字体不含中文，负号会显示成方框，所以关掉 unicode_minus，
       并且图上文字统一写英文（终端输出不受影响，可以用中文）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 必须早于 import pyplot
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False

# 跨图统一的配色：同一个模型在所有图里用同一种颜色，读者才能把不同图对照着看。
# 不能依赖 matplotlib 的默认色轮 —— 它按调用顺序分配，换个绘制顺序就变色。
TRAIN_COLOR = "#1f77b4"  # 蓝
VAL_COLOR = "#d62728"  # 红
COMPARE_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def save_figure(fig, out_path: str | Path) -> None:
    """存图并释放。用 fig.savefig 而不是 plt.savefig —— 后者存的是「当前活跃的
    figure」，有多个 figure 时容易存错。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"图已保存：{out_path}")


def plot_curves(
    history: dict,
    out_path: str | Path,
    title: str,
    *,
    target: float | None = None,
    target_label: str | None = None,
) -> None:
    """画单个模型的 Loss 曲线和准确率曲线（左 Loss、右 Accuracy）。

    target 是一条水平虚线，用来标出验收标准（Level 1 是 0.90，Level 2 是 0.96）。
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图：Loss 曲线
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train", color=TRAIN_COLOR)
    axes[0].plot(epochs, history["val_loss"], "s-", label="Validation", color=VAL_COLOR)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 右图：准确率曲线
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train", color=TRAIN_COLOR)
    axes[1].plot(epochs, history["val_acc"], "s-", label="Validation", color=VAL_COLOR)
    if target is not None:
        axes[1].axhline(target, ls="--", lw=1, color="gray",
                        label=target_label or f"{target:.0%} target")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # tight_layout 要放在 suptitle 之后，否则它是按「还没有总标题」的布局算边距的
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_path)


def plot_compare(
    histories: dict[str, dict],
    out_path: str | Path,
    title: str,
    *,
    metric: str = "val_acc",
    ylabel: str = "Validation accuracy",
    target: float | None = None,
    target_label: str | None = None,
) -> None:
    """把多个模型的同一条曲线画在一起（比如各模型的验证准确率）。

    histories 的键是模型名，值是 metrics json 里的 history 字典。
    颜色按 COMPARE_COLORS 的顺序固定分配，保证跨图一致。
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for i, (name, history) in enumerate(histories.items()):
        values = history[metric]
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, "o-", label=name, color=COMPARE_COLORS[i % len(COMPARE_COLORS)])

    if target is not None:
        ax.axhline(target, ls="--", lw=1, color="gray", label=target_label or f"{target:.0%} target")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_path)
