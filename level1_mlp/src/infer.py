"""Level 1 的单张图片推理脚本。

题目要求：编写脚本对单张图片进行推理。

两种用法：

    1. 拿测试集里的某张图（最常用，方便和对错答案对照）：
       python level1_mlp/src/infer.py --index 0

    2. 拿自己的图片文件（png/jpg，白底黑字或黑底白字都可以）：
       python level1_mlp/src/infer.py --image path/to/digit.png

推理和训练最大的区别在这里（也是容易搞错的地方）：

    model.train()  ->  Dropout 生效，随机丢神经元；用于训练
    model.eval()   ->  Dropout 关闭，所有神经元都用；用于推理

    再配合 with torch.no_grad(): 不建计算图，省显存也更快。
    如果推理时忘了 eval()，同一个输入两次跑出来的结果会不一样，
    因为 Dropout 每次随机丢的神经元不同。

输出的图包含三部分：原图、模型认为的答案、10 个类别各自的概率条形图。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无图形界面的环境下也能存图

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import build_model_from_config, count_parameters

# 脚本在 level1_mlp/src/ 下，往上两级就是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 必须和训练时完全一致：MNIST 全集的均值和标准差
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# matplotlib 默认字体不含中文，用英文标注避免出现一堆方框
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用训练好的 MLP 对单张图片做推理")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "mlp_mnist_best.pt"),
        help="模型权重路径（默认 checkpoints/mlp_mnist_best.pt）",
    )

    # 二选一：要么给测试集索引，要么给自己的图片
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", type=int, help="取 MNIST 测试集中的第几张图（从 0 开始）")
    source.add_argument("--image", type=str, help="自己的图片路径")

    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="结果图保存路径（默认存到 reports/samples/）",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: str) -> tuple[torch.nn.Module, dict]:
    """加载权重并重建模型。

    checkpoint 里除了权重，还存了模型结构参数（model.config()）和训练元信息，
    所以推理时不需要手动重复一遍网络结构——直接照着训练时的样子重建。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"找不到权重文件：{path}\n请先运行训练脚本：python level1_mlp/src/train.py"
        )

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model_from_config(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()  # 关闭 Dropout，固定住推理行为
    print(f"已加载权重：{path}")
    print(f"  训练时的 epoch      : {ckpt.get('epoch', '未知')}")
    print(f"  训练时的验证准确率  : {ckpt.get('val_acc', float('nan')):.2%}")
    print(f"  测试集准确率        : {ckpt.get('test_acc', float('nan')):.2%}")
    print(f"  参数量              : {count_parameters(model):,}")
    return model, ckpt


def load_image_from_dataset(index: int, data_dir: Path) -> tuple[torch.Tensor, int, Image.Image]:
    """从 MNIST 测试集里取一张图，返回（已归一化的张量, 真实标签, 原始 PIL 图）。"""
    from torchvision import datasets

    test_set = datasets.MNIST(root=str(data_dir), train=False, download=True)
    if not 0 <= index < len(test_set):
        raise IndexError(f"--index 需要在 0~{len(test_set) - 1} 之间，收到 {index}")

    pil_image, label = test_set[index]

    # 这里必须走和验证/测试时**一模一样**的预处理，否则输入分布和训练时不一致，
    # 预测结果会不可靠。eval() 里不能用随机增强，所以只有 ToTensor + Normalize。
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )
    tensor = transform(pil_image)
    return tensor, label, pil_image


def load_image_from_file(path: Path) -> tuple[torch.Tensor, None, Image.Image]:
    """加载自己的图片。

    做了三件事让它尽量贴近 MNIST 的格式：
      1. 转灰度（MNIST 是单通道）；
      2. 缩放到 28x28；
      3. 如果图片是「白底黑字」（手写照片常见），自动反色成 MNIST 的「黑底白字」。

    注意：这只做了最基本的对齐。MNIST 的字符是居中且归一化过的，
    自己拍的照片如果数字偏在角落或笔画太粗，准确率会明显下降，这是正常的。
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到图片：{path}")

    image = Image.open(path).convert("L")  # L = 8 位灰度

    # 判断是否需要反色：MNIST 的字体是白笔画 + 黑背景，
    # 所以平均像素值偏亮（偏白底）时说明是反的，需要翻过来。
    import numpy as np

    array = np.asarray(image, dtype="float32")
    if array.mean() > 127:  # 白底黑字
        image = Image.fromarray(255 - array.astype("uint8"))

    image = image.resize((28, 28), Image.BILINEAR)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )
    return transform(image), None, image


@torch.no_grad()
def predict(model: torch.nn.Module, tensor: torch.Tensor, device: str) -> torch.Tensor:
    """对单张图做一次前向计算，返回 10 个类别的概率。

    加了 batch 维度是因为模型要求输入是 (N, 1, 28, 28)，
    单张图本身是 (1, 28, 28)，用 unsqueeze(0) 补一个 N=1。
    """
    batch = tensor.unsqueeze(0).to(device)  # (1, 28, 28) -> (1, 1, 28, 28)
    logits = model(batch)                   # (1, 10) 原始打分
    probs = F.softmax(logits, dim=1)        # 打分 -> 概率（和为 1）
    return probs.squeeze(0).cpu()           # (10,)


def visualize(
    pil_image: Image.Image,
    probs: torch.Tensor,
    pred: int,
    true_label: int | None,
    out_path: Path,
) -> None:
    """把输入图片和预测概率画成一张图存下来。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 左图画的是**归一化之前**的原始像素。
    # 如果直接画归一化后的张量会出问题：像素值已被减去均值，有正有负，
    # matplotlib 把负值一律压成黑色，整张图看起来是一团糊，没法判断输入本身长什么样。
    axes[0].imshow(pil_image, cmap="gray")
    axes[0].set_title("Input (28x28)", fontsize=12)
    axes[0].axis("off")

    # 右图：10 个类别的概率
    colors = ["#d62728" if i == pred else "#aec7e8" for i in range(10)]
    axes[1].bar(range(10), probs.numpy(), color=colors)
    axes[1].set_xticks(range(10))
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Digit class", fontsize=11)
    axes[1].set_ylabel("Probability", fontsize=11)

    title = f"Prediction: {pred}  (confidence {probs[pred]:.2%})"
    if true_label is not None:
        title += f"\nGround truth: {true_label}  ->  {'CORRECT' if pred == true_label else 'WRONG'}"
    axes[1].set_title(title, fontsize=12)

    # 在每根柱子上标出具体数值，方便看清模型有多确定
    for i, p in enumerate(probs.tolist()):
        axes[1].text(i, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"结果图已保存：{out_path}")


def main() -> None:
    args = parse_args()

    model, ckpt = load_checkpoint(Path(args.checkpoint), args.device)

    # 取图：两个来源二选一
    if args.index is not None:
        tensor, true_label, pil_image = load_image_from_dataset(args.index, Path(args.data_dir))
        stem = f"test_{args.index:05d}"
    else:
        tensor, true_label, pil_image = load_image_from_file(Path(args.image))
        stem = Path(args.image).stem

    probs = predict(model, tensor, args.device)
    pred = int(probs.argmax().item())

    # 打印给终端看的完整结果
    print("\n" + "=" * 46)
    print(f"预测结果：{pred}")
    print(f"置信度  ：{probs[pred]:.4%}")
    if true_label is not None:
        print(f"真实标签：{true_label}")
        print(f"是否正确：{'✓ 正确' if pred == true_label else '✗ 错误'}")
    print("-" * 46)
    print("各类别概率：")
    for digit, p in enumerate(probs.tolist()):
        bar = "█" * int(p * 30)
        print(f"  {digit}: {p:7.4%}  {bar}")
    print("=" * 46)

    out_path = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "reports" / "samples" / f"mlp_infer_{stem}.png"
    )
    visualize(pil_image, probs, pred, true_label, out_path)


if __name__ == "__main__":
    main()
