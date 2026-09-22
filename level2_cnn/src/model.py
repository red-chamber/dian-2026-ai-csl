import torch
import torch.nn as nn

import sys
from pathlib import Path

# 脚本可能在任意工作目录下运行，先把仓库根补进 sys.path 才能 import common
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.utils import count_parameters  # noqa: F401  (重新导出，旧的 import 写法照旧可用)

__all__ = ["CNN", "count_parameters", "build_model_from_config"]


class CNN(nn.Module):
    """用于 MNIST 分类的卷积神经网络。

    structure(two conv layers + two FC layers):
          input                                  (N,  1, 28, 28)
          Conv2d( 1, 32, 3, padding=1) + ReLU    (N, 32, 28, 28)  卷积 1
          MaxPool2d(2)                           (N, 32, 14, 14)  池化
          Conv2d(32, 64, 3, padding=1) + ReLU    (N, 64, 14, 14)  卷积 2
          MaxPool2d(2)                           (N, 64,  7,  7)  池化
          Flatten                                (N, 3136)
          Dropout
          Linear(3136, 32) + ReLU                (N,   32)
          Dropout
          Linear(32, 10)                         (N,   10)   logits

    Args:
        conv_channels: 各卷积层的输出通道数。默认 (32, 64) 表示两层卷积
        fc_hidden:     展平后第一个全连接层的宽度
        in_channels:   输入通道数，MNIST 是灰度图所以是 1
        input_size:    输入图片的边长（28 表示 28x28）
        num_classes:   输出类别数
    """

    def __init__(
        self,
        conv_channels: tuple[int, ...] = (32, 64),
        fc_hidden: int = 32,
        in_channels: int = 1,
        input_size: int = 28,
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Saved for rebuild the model
        self.conv_channels = tuple(conv_channels)
        self.fc_hidden = fc_hidden
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_p = dropout

        # Conv part: Every layer consists of "Conv + ReLU + Pool"
        layers: list[nn.Module] = []
        prev_ch = in_channels
        for out_ch in self.conv_channels:
            # padding=1 且 kernel=3 时，卷积不改变边长（same padding）
            layers.append(nn.Conv2d(prev_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            # 池化把 H、W 各减半，相当于「下采样」，感受野随之翻倍
            layers.append(nn.MaxPool2d(kernel_size=2))
            prev_ch = out_ch
        self.features = nn.ModuleList(layers)

        # Calculate the flattened dimension automatically.
        spatial = input_size
        for _ in self.conv_channels:
            spatial //= 2
        flat_dim = prev_ch * spatial * spatial

        # 分类头 classification head
        self.flatten = nn.Flatten()                      # (N, 64, 7, 7) -> (N, 3136)
        self.drop = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(flat_dim, fc_hidden)        # 3136 -> 32
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc_hidden, num_classes)     # 32 -> 10

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize the weights

        Kaiming initialization
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Take a batch of images as input and output scores for each class.

        Args:
            x: 形状 (N, 1, 28, 28) 的图片批次
        Returns:
            形状 (N, 10) 的 logits
        """
        for layer in self.features:      # (N,  1, 28, 28) -> (N, 64, 7, 7)
            x = layer(x)
        x = self.flatten(x)              # (N, 64,  7,  7) -> (N, 3136)
        x = self.drop(x)
        x = self.act(self.fc1(x))        # (N, 3136) -> (N, 32)
        x = self.drop(x)
        return self.fc2(x)               # (N,   32) -> (N, 10)

    def config(self) -> dict:
        """保存 checkpoint 相关
        
        """
        return {
            "conv_channels": list(self.conv_channels),
            "fc_hidden": self.fc_hidden,
            "in_channels": self.in_channels,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "dropout": self.dropout_p,
        }


def build_model_from_config(config: dict) -> CNN:
    """Rebuild model
        the same as MLP
    """
    return CNN(
        conv_channels=tuple(config["conv_channels"]),
        fc_hidden=config.get("fc_hidden", 32),
        in_channels=config.get("in_channels", 1),
        input_size=config.get("input_size", 28),
        num_classes=config.get("num_classes", 10),
        dropout=config.get("dropout", 0.0),
    )


if __name__ == "__main__":
    model = CNN()
    print(model)
    print(f"\nTrainable parameter count: {count_parameters(model):,}")

    dummy = torch.randn(4, 1, 28, 28)
    print(f"\nShape of input tensor: {tuple(dummy.shape)}")

    with torch.no_grad():
        x = dummy
        for i, layer in enumerate(model.features):
            x = layer(x)
            print(f"  {type(layer).__name__:<12} -> {tuple(x.shape)}")
        flat = model.flatten(x)
        print(f"  {'Flatten':<12} -> {tuple(flat.shape)}   <- 64*7*7")
        out = model(dummy)
    print(f"Shape of output tensor: {tuple(out.shape)}(digit)")
