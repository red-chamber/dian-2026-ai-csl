"""Level 3：经典网络 AlexNet 与 ResNet

    两者本来是 224*224 的图片，被改造成 28*28，但主要结构不变：
    AlexNet   5 层 Conv + 3 层 fc，第一层用大卷积核，ReLU + Dropout，参数量集中在全连接层
    ResNet    3*3 小卷积核的残差块（BasicBlock），配合 BatchNorm 和跨层连接，用全局平均池化替换巨大的全连接层
"""

from __future__ import annotations

import torch
import torch.nn as nn
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.utils import count_parameters 


class AlexNet(nn.Module):
    """为 28*28 灰度图改造的 AlexNet

    Structure: 5 Conv + 3 fc

          input                                  (N, 1, 28, 28)
          Conv2d(  1,  64, 5, padding=2) + ReLU  (N, 64, 28, 28)
          MaxPool2d(2)                           (N, 64, 14, 14)
          Conv2d( 64, 192, 5, padding=2) + ReLU  (N, 192, 14, 14)
          MaxPool2d(2)                           (N, 192, 7, 7)
          Conv2d(192, 384, 3, padding=1) + ReL   (N, 384, 7, 7)
          Conv2d(384, 256, 3, padding=1) + ReLU  (N, 256, 7, 7)
          Conv2d(256, 256, 3, padding=1) + ReLU  (N, 256, 7, 7)
          MaxPool2d(2)                           (N, 256, 3, 3)
          Flatten + Dropout                      (N, 2304)
          Linear(2304, 1024) + ReLU + Dropot     (N, 1024)
          Linear(1024, 512) + ReLU               (N, 512)
          Linear(512, 10)                        (N, 10)

    Args:
        conv_channels: 各卷积层的输出通道数，AlexNet 原版 64/192/384/256/256
        fc_hidden:     两个隐藏全连接层的宽度
        in_channels:   输入通道数，灰度图 1
        input_size:    输入图片边长
        num_classes:   输出类别数
        dropout:       Dropout 概率
    """

    def __init__(
        self,
        conv_channels: tuple[int, ...] = (64, 192, 384, 256, 256),
        fc_hidden: tuple[int, ...] = (1024, 512),
        in_channels: int = 1,
        input_size: int = 28,
        num_classes: int = 10,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # Save in order to rebuild the model
        self.conv_channels = tuple(conv_channels)
        self.fc_hidden = tuple(fc_hidden)
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_p = dropout

        # 第 1、2 层之后各池化一次，最后一层之后再池化一次
        self.pool_after = tuple(sorted({0, 1, len(self.conv_channels) - 1}))

        # Conv part: 前两层用 5×5，后三层用 3×3。
        layers: list[nn.Module] = []
        prev_ch = in_channels
        for i, out_ch in enumerate(self.conv_channels):
            kernel = 5 if i < 2 else 3
            padding = kernel // 2  # kernel 为奇数时这样能维持 same padding
            layers.append(nn.Conv2d(prev_ch, out_ch, kernel_size=kernel, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            if i in self.pool_after:
                layers.append(nn.MaxPool2d(kernel_size=2))
            prev_ch = out_ch
        self.features = nn.ModuleList(layers)

        # Infer flattened dimensions 每次池化边长减半，卷积保持边长不变
        spatial = input_size
        for i in range(len(self.conv_channels)):
            if i in self.pool_after:
                spatial //= 2
        flat_dim = prev_ch * spatial * spatial

        # 分类头
        # flatten -> Dropout -> two fc -> output layer
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(p=dropout)
        self.classifier = nn.ModuleList()
        prev_dim = flat_dim
        for hidden_dim in self.fc_hidden:
            self.classifier.append(nn.Linear(prev_dim, hidden_dim))
            self.classifier.append(nn.ReLU(inplace=True))
            self.classifier.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim
        self.out = nn.Linear(prev_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        输入一批图片，输出每个类别的打分
        """
        for layer in self.features:
            x = layer(x)
        x = self.flatten(x)          # (N, 256, 3, 3) -> (N, 2304)
        x = self.drop(x)
        for layer in self.classifier:
            x = layer(x)             # (N, 2304) -> (N, 512)
        return self.out(x)           # (N, 512) -> (N, 10)

    def config(self) -> dict:
        return {
            "arch": "alexnet",
            "conv_channels": list(self.conv_channels),
            "fc_hidden": list(self.fc_hidden),
            "in_channels": self.in_channels,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "dropout": self.dropout_p,
        }


class BasicBlock(nn.Module):
    """基本残差块：两层 3*3 卷积 + 跨层连接。

        输入 x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> 跨层连接: 恒等/1*1卷积 -> ReLU -> 输出

    Args:
        in_channels:  输入通道数
        out_channels: 输出通道数
        stride:       步长
        residual:     是否使用跨层连接。设为 False 就是一个普通的卷积块，used for PlainNet（消融实验）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跨层连接：形状能和主路径直接相加时就恒等映射，否则用 1×1 卷积调整
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x) if self.residual else 0

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        return self.relu(out)


class ResNet18(nn.Module):
    """为 28*28 灰度图改造的 ResNet-18

    1 个 stem + 4 个 stage，每个 stage 2 个残差块: 

          input                                    (N,   1, 28, 28)   没有 MaxPool，因为 28*28 太小了
          Conv2d(1, 64, 3, padding=1) + BN + ReLU  (N,  64, 28, 28)   即 stem
          stage1  2 BasicBlock( 64 ->  64)         (N,  64, 28, 28)
          stage2  2 BasicBlock( 64 -> 128, s=2)    (N, 128, 14, 14)
          stage3  2 BasicBlock(128 -> 256, s=2)    (N, 256,  7,  7)
          stage4  2 BasicBlock(256 -> 512, s=2)    (N, 512,  4,  4)
          AdaptiveAvgPool2d(1)                     (N, 512,  1,  1)
          Flatten                                  (N, 512)
          Linear(512, 10)                          (N,  10)

    Args:
        num_blocks: 每个 stage 的残差块数量
        widths:     各 stage 的输出通道数，默认 (64,128,256,512)
        residual:   是否使用残差连接
    """

    def __init__(
        self,
        num_blocks: tuple[int, ...] = (2, 2, 2, 2),
        widths: tuple[int, ...] = (64, 128, 256, 512),
        in_channels: int = 1,
        input_size: int = 28,
        num_classes: int = 10,
        residual: bool = True,
    ) -> None:
        super().__init__()

        self.num_blocks = tuple(num_blocks)
        self.widths = tuple(widths)
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_classes = num_classes
        self.residual = residual

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.widths[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.widths[0]),
            nn.ReLU(inplace=True),
        )

        # 4 个 stage
        stages: list[nn.Module] = []
        prev_ch = self.widths[0]
        for stage_i, (out_ch, n_blocks) in enumerate(zip(self.widths, self.num_blocks)):
            for block_i in range(n_blocks):
                stride = 2 if (stage_i > 0 and block_i == 0) else 1
                stages.append(BasicBlock(prev_ch, out_ch, stride=stride, residual=residual))
                prev_ch = out_ch
        self.stages = nn.ModuleList(stages)

        # 全局平均池化、单层全连接
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(prev_ch, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """卷积层用 Kaiming 初始化，BatchNorm 用默认值（weight=1, bias=0）
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        """
        x = self.stem(x)
        for block in self.stages:
            x = block(x)
        x = self.pool(x)         # (N, 512, 4, 4) -> (N, 512, 1, 1)
        x = self.flatten(x)      # (N, 512)
        return self.fc(x)        # (N, 512) -> (N, 10)

    def config(self) -> dict:
        return {
            "arch": "resnet",
            "num_blocks": list(self.num_blocks),
            "widths": list(self.widths),
            "in_channels": self.in_channels,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "residual": self.residual,
        }


# 供命令行 --model 选择用
ARCHITECTURES: dict[str, type[nn.Module]] = {
    "alexnet": AlexNet,
    "resnet": ResNet18,
}


def build_model_from_config(config: dict) -> nn.Module:
    """用 ckpt 的参数重建模型
    
    """
    arch = config.get("arch", "alexnet")
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown Model {arch!r},Canditates: {sorted(ARCHITECTURES)}")

    if arch == "alexnet":
        return AlexNet(
            conv_channels=tuple(config["conv_channels"]),
            fc_hidden=tuple(config.get("fc_hidden", (1024, 512))),
            in_channels=config.get("in_channels", 1),
            input_size=config.get("input_size", 28),
            num_classes=config.get("num_classes", 10),
            dropout=config.get("dropout", 0.0),
        )

    return ResNet18(
        num_blocks=tuple(config.get("num_blocks", (2, 2, 2, 2))),
        widths=tuple(config.get("widths", (64, 128, 256, 512))),
        in_channels=config.get("in_channels", 1),
        input_size=config.get("input_size", 28),
        num_classes=config.get("num_classes", 10),
        residual=config.get("residual", True),
    )


if __name__ == "__main__":
    dummy = torch.randn(4, 1, 28, 28)
    print(f"Input shape: {tuple(dummy.shape)}\n")

    for name, build in (
        ("AlexNet", lambda: AlexNet()),
        ("ResNet18", lambda: ResNet18()),
        ("PlainNet18（without residuals）", lambda: ResNet18(residual=False)),
    ):
        model = build()
        model.eval()
        print("=" * 62)
        print(f"{name}   Trainable parameter count: {count_parameters(model):,}")
        print("=" * 62)
        with torch.no_grad():
            out = model(dummy)
        print(f"Output shape: {tuple(out.shape)}\n")
