"""权重的存与读。

checkpoint 里存的不只是权重，还有重建模型所需的结构参数（model_config）
和训练元信息，所以推理时不需要手写网络结构 —— 直接照着训练时的样子重建。

    model_state    权重
    model_config   结构参数，来自 model.config()
    epoch/val_acc  存下来时的最优轮次与验证准确率
    args           当时的命令行配置
    test_acc       训练全部结束后补写（中途被打断的 checkpoint 里没有这个键）
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .utils import count_parameters


def save_checkpoint(path: str | Path, *, model: nn.Module, model_config: dict, **extra) -> Path:
    """把权重、结构参数和若干元信息写成一个 .pt 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "model_config": model_config, **extra},
        path,
    )
    return path


def load_weights(path: str | Path, device: str, build_fn) -> tuple[nn.Module, dict]:
    """加载权重并重建模型，不做任何打印。

    需要自定义输出内容的调用方（比如 Level 4 要报 PSNR / SSIM 而不是准确率）
    用这个；只想直接看模型信息的用下面的 `load_checkpoint`。

    build_fn 是各 Level 自己的 `build_model_from_config`，由调用方传入 ——
    这样本模块不需要知道模型长什么样，MLP / CNN / AlexNet / ResNet / U-Net
    共用同一段逻辑。

    权重文件里存的张量记着它原本所在的设备，所以不带 map_location 时，
    GPU 上存的权重在纯 CPU 机器上会加载失败。map_location 解决的是
    「文件能不能读出来」，和后面 model.to(device) 的「模型住在哪」是两件事，
    两个都不能省：load_state_dict 只拷贝数据，不改变目标参数的设备。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find the weights file: {path}\nPlease run the training script first."
        )

    # weights_only=False 关掉 2.6 起收紧的 pickle 限制：本文件里除了张量还存了
    # args 等普通对象。代价是只应该加载自己训练出来的文件。
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_fn(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()  # Switch to inference mode.(Disable Dropout)
    return model, ckpt


def load_checkpoint(path: str | Path, device: str, build_fn) -> tuple[nn.Module, dict]:
    """加载权重并重建模型，并打印分类任务的模型信息（Level 1-3 用）。

    具体的加载逻辑在 `load_weights` 里，这里只负责打印。
    """
    model, ckpt = load_weights(path, device, build_fn)

    print(f"Path to loaded weights: {path}")
    print(f"Training epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Validation accuracy during training: {ckpt.get('val_acc', float('nan')):.2%}")
    print(f"Accuracy of test set: {ckpt.get('test_acc', float('nan')):.2%}")
    print(f"Parameter count: {count_parameters(model):,}")
    return model, ckpt
