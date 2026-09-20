"""单张图片推理脚本

Dropout 关闭，所有神经元都用
with torch.no_grad() 不建计算图，省显存

outputs：原图、模型输出、10 个类别各自的概率条形图
"""

from __future__ import annotations

import argparse
from pathlib import Path

# about plotting graphs
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import build_model_from_config, count_parameters

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    '''Define the command-line interface of the program.

    '''
    parser = argparse.ArgumentParser(description="Run inference on a single image using an MLP.") # 创建解析器
    parser.add_argument(
        "--checkpoint",                                                  # 可选参数
        type=str,                                                        # 全部转成字符串
        default=str(PROJECT_ROOT / "checkpoints" / "mlp_mnist_best.pt"),
        help="Model weights path",
    )

    # Select an image in test set or a custom image.
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", type=int, help="Fetch the n-th image from the MNIST test set (0-indexed)")
    source.add_argument("--image", type=str, help="Path to your cunstom image.")

    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output figure save path",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: str) -> tuple[torch.nn.Module, dict]:
    """Load weights and reconstruct the model from checkpoint.
    
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find the weights file: {path}\nPlease run the training script first."
        )

    ckpt = torch.load(path, map_location=device, weights_only=False) # standard practice for robustness
    # Rebuild the model structure and load weights.
    model = build_model_from_config(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()  # Switch to inference mode.(Disable Dropout)
    print(f"Path to loaded weights: {path}")
    print(f"Training epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Validation accuracy during training: {ckpt.get('val_acc', float('nan')):.2%}") # default value: nan 验证准确率
    print(f"Accuracy of test set: {ckpt.get('test_acc', float('nan')):.2%}")
    print(f"Parameter count: {count_parameters(model):,}")
    return model, ckpt


def load_image_from_dataset(index: int, data_dir: Path) -> tuple[torch.Tensor, int, Image.Image]:
    """Fetch one image from test set and
        return (normalized tensor, ground-truth label, original PIL image).
        Perhaps the most critical part...
    """
    from torchvision import datasets

    test_set = datasets.MNIST(root=str(data_dir), train=False, download=True)
    if not 0 <= index < len(test_set):
        raise IndexError(f"--index needs to be from 0 to {len(test_set) - 1}")

    pil_image, label = test_set[index]

    # Convert to tensor and normalize.(WITHOUT other process)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )
    tensor = transform(pil_image)
    return tensor, label, pil_image


def load_image_from_file(path: Path) -> tuple[torch.Tensor, None, Image.Image]:
    """Load your custom image.

    What has been done to make it as close to the MNIST format as possible?
      1. Convert to greyscale image.
      2. Resize to 28*28.
      3. Convert to black background with white digits(if the given image performs the opposite).
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot find the image: {path}")

    image = Image.open(path).convert("L")  # Convert to 8-bit grayscale image.
    
    import numpy as np

    array = np.asarray(image, dtype="float32")
    if array.mean() > 127: # If the average pixel value greater than 127,the image has a white background.
        image = Image.fromarray(255 - array.astype("uint8"))

    image = image.resize((28, 28), Image.BILINEAR)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )
    return transform(image), None, image


@torch.no_grad()
def predict(model: torch.nn.Module, tensor: torch.Tensor, device: str) -> torch.Tensor:
    """对单张图做一次前向计算，返回 10 个类别的概率

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
    """Plot the input image and the predicted probabilities in one figure.
    
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # The left figure displays the pixels before normalizing.
    axes[0].imshow(pil_image, cmap="gray")
    axes[0].set_title("Input (28x28)", fontsize=12)
    axes[0].axis("off")

    # The right figure shows the probabilities.
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

    for i, p in enumerate(probs.tolist()):
        axes[1].text(i, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Result figure is saved in {out_path}")


def main() -> None:
    args = parse_args()

    model, ckpt = load_checkpoint(Path(args.checkpoint), args.device)

    # Try to fetch an image.
    if args.index is not None:
        tensor, true_label, pil_image = load_image_from_dataset(args.index, Path(args.data_dir))
        stem = f"test_{args.index:05d}"
    else:
        tensor, true_label, pil_image = load_image_from_file(Path(args.image))
        stem = Path(args.image).stem

    probs = predict(model, tensor, args.device)
    pred = int(probs.argmax().item())

    print("\n" + "=" * 46)
    print(f"Predicted results: {pred}")
    print(f"Confidence: {probs[pred]:.4%}")
    if true_label is not None:
        print(f"Ground-true label: {true_label}")
        print(f"{'Right' if pred == true_label else 'Wrong'}")
    print("-" * 46)
    print("Possibilities of every class: ")
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
