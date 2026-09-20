"""MLP vs CNN 四角度对比脚本（Level 2 的核心产物）

验收标准要求「从准确率 / 参数量 / 收敛速度 / 错误样本四个角度与 MLP 对比」，
本脚本把这四项一次跑完，产出可直接写进 README 的数字与图。

    python level2_cnn/src/compare.py

前提：两个模型都已训练过，即下面两组文件同时存在
    checkpoints/mlp_mnist_best.pt   +  reports/metrics/mlp_mnist.json    （Level 1）
    checkpoints/cnn_mnist_best.pt   +  reports/metrics/cnn_mnist.json    （Level 2）

outputs:
    reports/figures/mlp_vs_cnn_curves.png      收敛速度对比（Loss / Accuracy 双曲线）
    reports/figures/mlp_vs_cnn_confusion.png   两个混淆矩阵并排
    reports/samples/mlp_vs_cnn_errors.png      错误样本分类网格
    reports/metrics/compare_mlp_cnn.json       全部对比数字

四个角度的设计说明：

    准确率   两个模型在同一份官方测试集（10000 张）上各跑一遍，口径完全一致
    参数量   用同一个 count_parameters()，累加 numel()，不区分层类型
    收敛速度 直接读各自 metrics json 里的逐轮 history，不重新训练
    错误样本 把测试集样本分成「都错 / 只有 MLP 错 / 只有 CNN 错」三类，
             并统计各自最容易混淆的数字对 —— 这是最能说明问题的一项
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import count_parameters

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEVEL1_SRC = PROJECT_ROOT / "level1_mlp" / "src"

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def _load_level1_model_module():
    """加载 Level 1 的 model.py。

    两个 Level 的模型文件都叫 model.py，直接 `import model` 会撞名
    （脚本所在目录的 model.py 会被优先找到），所以用 importlib 按文件路径
    把它加载成一个名字不同的独立模块。
    """
    path = LEVEL1_SRC / "model.py"
    if not path.exists():
        raise FileNotFoundError(f"找不到 Level 1 的模型定义：{path}")
    spec = importlib.util.spec_from_file_location("level1_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="对比 MLP 与 CNN 在 MNIST 上的表现")
    p.add_argument("--mlp-ckpt", type=str, default=str(PROJECT_ROOT / "checkpoints" / "mlp_mnist_best.pt"))
    p.add_argument("--cnn-ckpt", type=str, default=str(PROJECT_ROOT / "checkpoints" / "cnn_mnist_best.pt"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "raw"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--grid-n", type=int, default=6, help="错误样本图里每类最多展示几张")
    p.add_argument("--top-k", type=int, default=5, help="统计前 K 个最容易混淆的数字对")
    return p.parse_args()


def load_model(ckpt_path: Path, builder, device: str):
    """按 checkpoint 重建模型并加载权重，返回 (model, ckpt)。"""
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"找不到权重文件：{ckpt_path}\n请先训练对应模型（train.py）再运行对比脚本。"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = builder(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def load_metrics(tag: str) -> dict:
    """读取训练时导出的 metrics json（用于取逐轮 history）。"""
    path = PROJECT_ROOT / "reports" / "metrics" / f"{tag}.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到实验指标：{path}\n请先训练对应模型（train.py）。")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def collect_predictions(model, loader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在测试集上跑一遍，返回 (预测标签, 真实标签, 预测置信度)。

    DataLoader 用 shuffle=False，所以数组下标和数据集下标一一对应，
    可以直接拿来定位具体是哪张图。
    """
    preds, trues, confs = [], [], []
    for images, labels in loader:
        probs = F.softmax(model(images.to(device)), dim=1)
        conf, pred = probs.max(dim=1)
        preds.append(pred.cpu())
        confs.append(conf.cpu())
        trues.append(labels)
    return (
        torch.cat(preds).numpy(),
        torch.cat(trues).numpy(),
        torch.cat(confs).numpy(),
    )


def confusion_matrix(true: np.ndarray, pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """手算混淆矩阵，避免为了一个函数引入 scikit-learn 依赖。

    cm[i, j] = 真实是 i、却被预测成 j 的张数。
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(cm, (true, pred), 1)
    return cm


def top_confusions(cm: np.ndarray, k: int = 5) -> list[tuple[int, int, int]]:
    """取最容易混淆的 (真实, 预测, 次数) 前 k 项（只看错分的格子）。"""
    pairs = [
        (i, j, int(cm[i, j]))
        for i in range(cm.shape[0])
        for j in range(cm.shape[1])
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda x: -x[2])
    return pairs[:k]


def epochs_to_reach(history: dict, threshold: float) -> int | None:
    """第一次达到某个验证准确率是在第几轮；达不到返回 None。"""
    for i, acc in enumerate(history["val_acc"], start=1):
        if acc >= threshold:
            return i
    return None


def plot_convergence(mlp_hist: dict, cnn_hist: dict, out_path: Path, mlp_name: str, cnn_name: str) -> None:
    """收敛速度对比：Loss 与 Accuracy 两组曲线，MLP / CNN 各一条。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlp_epochs = range(1, len(mlp_hist["val_acc"]) + 1)
    cnn_epochs = range(1, len(cnn_hist["val_acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图：验证损失
    axes[0].plot(mlp_epochs, mlp_hist["val_loss"], "o-", color="#1f77b4", label=f"MLP ({mlp_name})")
    axes[0].plot(cnn_epochs, cnn_hist["val_loss"], "s-", color="#2ca02c", label=f"CNN ({cnn_name})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("Validation loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 右图：验证准确率
    axes[1].plot(mlp_epochs, mlp_hist["val_acc"], "o-", color="#1f77b4", label=f"MLP ({mlp_name})")
    axes[1].plot(cnn_epochs, cnn_hist["val_acc"], "s-", color="#2ca02c", label=f"CNN ({cnn_name})")
    axes[1].axhline(0.96, ls="--", lw=1, color="gray", label="96% target")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy")
    axes[1].set_title("Validation accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Convergence: MLP vs CNN on MNIST", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"收敛曲线已保存：{out_path}")


def plot_confusions(
    mlp_cm: np.ndarray, cnn_cm: np.ndarray, out_path: Path, mlp_name: str, cnn_name: str
) -> None:
    """两个混淆矩阵并排。行 = 真实类别，列 = 预测类别。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    for ax, cm, name in ((axes[0], mlp_cm, mlp_name), (axes[1], cnn_cm, cnn_name)):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}  (errors: {int(cm.sum() - np.trace(cm))})")
        # 每格写上数字；错误格用红色标出，方便一眼看出错在哪
        for i in range(10):
            for j in range(10):
                if cm[i, j] == 0:
                    continue
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="#d62728" if i != j else "#333333",
                    fontweight="bold" if i != j else "normal",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion matrices (red = misclassified)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"混淆矩阵已保存：{out_path}")


def plot_error_grid(
    raw_dataset,
    true: np.ndarray,
    mlp_pred: np.ndarray,
    cnn_pred: np.ndarray,
    out_path: Path,
    per_class: int = 6,
) -> dict:
    """把错误样本分成三类并画成网格图。

    三列分别是：
        两者都错   —— 两个模型都搞不定的「硬样本」
        只有 MLP 错 —— CNN 修好了的样本（卷积带来的收益）
        只有 CNN 错 —— MLP 对而 CNN 错的样本（用来审视 CNN 的短板）
    """
    mlp_err = mlp_pred != true
    cnn_err = cnn_pred != true

    groups = [
        ("Both wrong", np.where(mlp_err & cnn_err)[0]),
        ("Only MLP wrong", np.where(mlp_err & ~cnn_err)[0]),
        ("Only CNN wrong", np.where(~mlp_err & cnn_err)[0]),
    ]

    rows = min(per_class, max((len(idx) for _, idx in groups), default=0))
    if rows == 0:
        print("两个模型在测试集上都没有错分样本，跳过错误样本图。")
        return {"both_wrong": 0, "only_mlp_wrong": 0, "only_cnn_wrong": 0}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, 3, figsize=(7.2, rows * 2.3), squeeze=False)

    for col, (title, idx) in enumerate(groups):
        axes[0][col].set_title(f"{title}\n(n={len(idx)})", fontsize=11)
        for row in range(rows):
            ax = axes[row][col]
            ax.axis("off")
            if row >= len(idx):
                continue
            i = int(idx[row])
            pil_image, _ = raw_dataset[i]           # 用未归一化的原图来显示
            ax.imshow(pil_image, cmap="gray")
            ax.set_title(
                f"true={true[i]}  MLP={mlp_pred[i]}  CNN={cnn_pred[i]}",
                fontsize=7.5,
            )

    fig.suptitle("Error samples: where CNN wins and where it loses", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"错误样本图已保存：{out_path}")

    return {
        "both_wrong": int((mlp_err & cnn_err).sum()),
        "only_mlp_wrong": int((mlp_err & ~cnn_err).sum()),
        "only_cnn_wrong": int((~mlp_err & cnn_err).sum()),
    }


def main() -> None:
    args = parse_args()
    device = args.device

    print("=" * 66)
    print("Level 2：MLP vs CNN 四角度对比")
    print("=" * 66)

    # ---------- 加载两个模型 ----------
    level1 = _load_level1_model_module()
    mlp, mlp_ckpt = load_model(Path(args.mlp_ckpt), level1.build_model_from_config, device)
    cnn, cnn_ckpt = load_model(Path(args.cnn_ckpt), load_cnn_builder(), device)

    mlp_name = f"MLP {mlp_ckpt['model_config']['hidden_sizes']}"
    cnn_name = f"CNN {cnn_ckpt['model_config']['conv_channels']}"
    print(f"MLP : {mlp_name}")
    print(f"CNN : {cnn_name}")

    mlp_metrics = load_metrics("mlp_mnist")
    cnn_metrics = load_metrics("cnn_mnist")

    # ---------- 测试集：一份带归一化喂模型，一份原图用于显示 ----------
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )
    data_dir = Path(args.data_dir)
    test_set = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)
    raw_test = datasets.MNIST(root=str(data_dir), train=False, download=True)  # 不做变换，取 PIL 原图
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device == "cuda"),
    )
    print(f"\n测试集: {len(test_set):,} 张（官方 test split）\n")

    # ---------- 角度 1：参数量 ----------
    mlp_params = count_parameters(mlp)
    cnn_params = count_parameters(cnn)

    # ---------- 角度 2：准确率（两模型在同一份测试集上各跑一遍）----------
    mlp_pred, true, mlp_conf = collect_predictions(mlp, test_loader, device)
    cnn_pred, _, cnn_conf = collect_predictions(cnn, test_loader, device)
    mlp_acc = float((mlp_pred == true).mean())
    cnn_acc = float((cnn_pred == true).mean())

    # ---------- 角度 3：收敛速度（读逐轮 history）----------
    mlp_reach = epochs_to_reach(mlp_metrics["history"], 0.96)
    cnn_reach = epochs_to_reach(cnn_metrics["history"], 0.96)
    mlp_sec_per_epoch = mlp_metrics["train_time_sec"] / mlp_metrics["epochs"]
    cnn_sec_per_epoch = cnn_metrics["train_time_sec"] / cnn_metrics["epochs"]

    # ---------- 角度 4：错误样本 ----------
    mlp_cm = confusion_matrix(true, mlp_pred)
    cnn_cm = confusion_matrix(true, cnn_pred)
    groups = plot_error_grid(raw_test, true, mlp_pred, cnn_pred,
                             PROJECT_ROOT / "reports" / "samples" / "mlp_vs_cnn_errors.png",
                             per_class=args.grid_n)

    plot_convergence(mlp_metrics["history"], cnn_metrics["history"],
                     PROJECT_ROOT / "reports" / "figures" / "mlp_vs_cnn_curves.png",
                     mlp_name, cnn_name)
    plot_confusions(mlp_cm, cnn_cm,
                    PROJECT_ROOT / "reports" / "figures" / "mlp_vs_cnn_confusion.png",
                    mlp_name, cnn_name)

    # ---------- 汇总输出 ----------
    def fmt_epoch(e: int | None) -> str:
        return f"第 {e} 轮" if e is not None else "未达到"

    print("\n" + "=" * 66)
    print("角度 1｜参数量")
    print("=" * 66)
    print(f"  {mlp_name:<22}: {mlp_params:>9,}")
    print(f"  {cnn_name:<22}: {cnn_params:>9,}")
    print(f"  CNN 只有 MLP 的 {cnn_params / mlp_params:.1%}")

    print("\n" + "=" * 66)
    print("角度 2｜准确率（官方测试集 10000 张）")
    print("=" * 66)
    print(f"  {mlp_name:<22}: {mlp_acc:>9.2%}   错误 {int((mlp_pred != true).sum()):>4} 张")
    print(f"  {cnn_name:<22}: {cnn_acc:>9.2%}   错误 {int((cnn_pred != true).sum()):>4} 张")
    print(f"  准确率提升              : {(cnn_acc - mlp_acc) * 100:>+9.2f} 个百分点")

    print("\n" + "=" * 66)
    print("角度 3｜收敛速度")
    print("=" * 66)
    print(f"  验证准确率达到 96%      : MLP {fmt_epoch(mlp_reach)} / CNN {fmt_epoch(cnn_reach)}")
    print(f"  第 1 轮验证准确率       : MLP {mlp_metrics['history']['val_acc'][0]:.2%} / "
          f"CNN {cnn_metrics['history']['val_acc'][0]:.2%}")
    print(f"  每轮平均耗时            : MLP {mlp_sec_per_epoch:.1f}s / CNN {cnn_sec_per_epoch:.1f}s")
    print(f"  最优 epoch / 验证准确率 : MLP 第 {mlp_metrics['best_epoch']} 轮 "
          f"{mlp_metrics['best_val_acc']:.2%} / CNN 第 {cnn_metrics['best_epoch']} 轮 "
          f"{cnn_metrics['best_val_acc']:.2%}")

    print("\n" + "=" * 66)
    print("角度 4｜错误样本")
    print("=" * 66)
    print(f"  两者都错                : {groups['both_wrong']:>4} 张")
    print(f"  只有 MLP 错（CNN 修好） : {groups['only_mlp_wrong']:>4} 张")
    print(f"  只有 CNN 错（MLP 对）   : {groups['only_cnn_wrong']:>4} 张")
    print(f"\n  {mlp_name} 最容易混淆的数字对：")
    for t, p, n in top_confusions(mlp_cm, args.top_k):
        print(f"    真实 {t} 被认成 {p} : {n:>3} 次")
    print(f"\n  {cnn_name} 最容易混淆的数字对：")
    for t, p, n in top_confusions(cnn_cm, args.top_k):
        print(f"    真实 {t} 被认成 {p} : {n:>3} 次")

    # 置信度：错的时候模型有多"自信"，反映它是否知道自己在犯错
    print(f"\n  错分样本的平均置信度    : MLP {mlp_conf[mlp_pred != true].mean():.2%} / "
          f"CNN {cnn_conf[cnn_pred != true].mean():.2%}")
    print("=" * 66)

    # ---------- 存 JSON ----------
    result = {
        "mlp": {"name": mlp_name, "params": mlp_params, "test_acc": mlp_acc,
                "test_errors": int((mlp_pred != true).sum()),
                "best_epoch": mlp_metrics["best_epoch"],
                "best_val_acc": mlp_metrics["best_val_acc"],
                "sec_per_epoch": round(mlp_sec_per_epoch, 2),
                "epoch_to_96": mlp_reach,
                "top_confusions": top_confusions(mlp_cm, args.top_k)},
        "cnn": {"name": cnn_name, "params": cnn_params, "test_acc": cnn_acc,
                "test_errors": int((cnn_pred != true).sum()),
                "best_epoch": cnn_metrics["best_epoch"],
                "best_val_acc": cnn_metrics["best_val_acc"],
                "sec_per_epoch": round(cnn_sec_per_epoch, 2),
                "epoch_to_96": cnn_reach,
                "top_confusions": top_confusions(cnn_cm, args.top_k)},
        "error_groups": groups,
        "figures": {
            "curves": "reports/figures/mlp_vs_cnn_curves.png",
            "confusion": "reports/figures/mlp_vs_cnn_confusion.png",
            "errors": "reports/samples/mlp_vs_cnn_errors.png",
        },
    }
    out_path = PROJECT_ROOT / "reports" / "metrics" / "compare_mlp_cnn.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存：{out_path}")


def load_cnn_builder():
    """返回 CNN 的重建函数。

    单独包一层是为了在 main 里和 Level 1 的 builder 对称地传参，
    同时避免在模块顶层就 import 本目录的 model（那样会和 Level 1 的 model.py 撞名）。
    """
    from model import build_model_from_config

    return build_model_from_config


if __name__ == "__main__":
    main()
