"""单张图片推理脚本（Level 3：AlexNet / ResNet）

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无图形界面的环境下也能存图
import matplotlib.pyplot as plt  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.checkpoint import load_checkpoint  # noqa: E402
from common.data import get_dataset_spec  # noqa: E402
from common.utils import PROJECT_ROOT  # noqa: E402
from model import build_model_from_config  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    """Define the command-line interface of the program."""
    parser = argparse.ArgumentParser(description="Run inference on a single image using AlexNet / ResNet.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "resnet_fashionmnist_best.pt"),
        help="Model weights path",
    )
    parser.add_argument("--dataset", type=str, default="fashion-mnist",
                        choices=["mnist", "fashion-mnist"], help="")

    # Select an image in test set or a custom image.
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", type=int, help="Fetch the n-th image from the test set (0-indexed)")
    source.add_argument("--image", type=str, help="Path to your custom image.")

    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None, help="output figure save path")
    return parser.parse_args()


def load_image_from_dataset(index: int, spec: dict, data_dir: Path) -> tuple[torch.Tensor, int, Image.Image]:
    """Fetch one image from test set and
        return (normalized tensor, ground-truth label, original PIL image).
    """
    test_set = spec["cls"](root=str(data_dir), train=False, download=True)
    if not 0 <= index < len(test_set):
        raise IndexError(f"--index needs to be from 0 to {len(test_set) - 1}")

    pil_image, label = test_set[index]

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((spec["mean"],), (spec["std"],))]
    )
    return transform(pil_image), label, pil_image


def load_image_from_file(path: Path, spec: dict) -> tuple[torch.Tensor, None, Image.Image]:
    """Load your custom image.

    What has been done to make it as close to the dataset format as possible?
      1. Convert to greyscale image.
      2. Resize to 28*28.
      3. Convert to black background with white object (if the given image is the opposite).
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot find the image: {path}")

    image = Image.open(path).convert("L")  # Convert to 8-bit grayscale image.

    import numpy as np

    array = np.asarray(image, dtype="float32")
    if array.mean() > 127:  # White background -> invert to match the datasets
        image = Image.fromarray(255 - array.astype("uint8"))

    image = image.resize((spec["input_size"], spec["input_size"]), Image.BILINEAR)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((spec["mean"],), (spec["std"],))]
    )
    return transform(image), None, image


@torch.no_grad()
def predict(model: torch.nn.Module, tensor: torch.Tensor, device: str) -> torch.Tensor:
    """对单张图做一次前向计算，返回 10 个类别的概率"""
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
    classes: list[str],
) -> None:
    """Plot the input image and the predicted probabilities in one figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # The left figure displays the pixels before normalizing.
    axes[0].imshow(pil_image, cmap="gray")
    axes[0].set_title("Input (28x28)", fontsize=12)
    axes[0].axis("off")

    # The right figure shows the probabilities. 
    colors = ["#d62728" if i == pred else "#aec7e8" for i in range(len(classes))]
    axes[1].barh(range(len(classes)), probs.numpy(), color=colors)
    axes[1].set_yticks(range(len(classes)))
    axes[1].set_yticklabels(classes, fontsize=8)
    axes[1].invert_yaxis() 
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Probability", fontsize=10)

    title = f"Prediction: {classes[pred]}  (confidence {probs[pred]:.2%})"
    if true_label is not None:
        ok = "CORRECT" if pred == true_label else "WRONG"
        title += f"\nGround truth: {classes[true_label]}  ->  {ok}"
    axes[1].set_title(title, fontsize=11)

    for i, p in enumerate(probs.tolist()):
        axes[1].text(p + 0.01, i, f"{p:.2f}", va="center", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Result figure is saved in {out_path}")


def main() -> None:
    args = parse_args()
    spec = get_dataset_spec(args.dataset)
    classes = spec["classes"]

    model, ckpt = load_checkpoint(Path(args.checkpoint), args.device, build_model_from_config)

    # Try to fetch an image.
    if args.index is not None:
        tensor, true_label, pil_image = load_image_from_dataset(args.index, spec, Path(args.data_dir))
        stem = f"test_{args.index:05d}"
    else:
        tensor, true_label, pil_image = load_image_from_file(Path(args.image), spec)
        stem = Path(args.image).stem

    probs = predict(model, tensor, args.device)
    pred = int(probs.argmax().item())

    print("\n" + "=" * 46)
    print(f"Predicted results: {classes[pred]}")
    print(f"Confidence: {probs[pred]:.4%}")
    if true_label is not None:
        print(f"Ground-true label: {classes[true_label]}")
        print(f"{'Right' if pred == true_label else 'Wrong'}")
    print("-" * 46)
    print("Possibilities of every class: ")
    for class_id, p in enumerate(probs.tolist()):
        bar = "#" * int(p * 30)
        print(f"  {classes[class_id]:<12} {p:7.4%}  {bar}")
    print("=" * 46)

    model_tag = args.checkpoint.split("/")[-1].split("_")[0] if "/" in args.checkpoint else "model"
    out_path = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "reports" / "samples" / f"{model_tag}_infer_{stem}.png"
    )
    visualize(pil_image, probs, pred, true_label, out_path, classes)


if __name__ == "__main__":
    main()
