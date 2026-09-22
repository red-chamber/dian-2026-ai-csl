"""Level 3：经典网络 AlexNet 与 ResNet

    两者本来是面向 224*224 的图片，被改造成面向 MNIST/Fashion MNIST 28*28，但主要结构不变：
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
        fc_hidden:     两个隐藏全连接层的宽度，对应原版的 4096/4096（这里缩小了）
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

        # 池化节奏：第 1、2 层之后各池化一次，最后一层之后再池化一次，
        # 于是 28 -> 14 -> 7 -> 3。中间两层尽量保持分辨率，这是原版的做法。
        self.pool_after = tuple(sorted({0, 1, len(self.conv_channels) - 1}))

        # 卷积部分。前两层用 5×5（原版是 11×11 / 5×5），后三层用 3×3。
        layers: list[nn.Module] = []
        prev_ch = in_channels
        for i, out_ch in enumerate(self.conv_channels):
            kernel = 5 if i < 2 else 3
            padding = kernel // 2  # kernel 为奇数时这样能保持边长（same padding）
            layers.append(nn.Conv2d(prev_ch, out_ch, kernel_size=kernel, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            if i in self.pool_after:
                layers.append(nn.MaxPool2d(kernel_size=2))
            prev_ch = out_ch
        self.features = nn.ModuleList(layers)

        # 推算展平后的维度：每次池化边长减半，卷积保持边长不变
        spatial = input_size
        for i in range(len(self.conv_channels)):
            if i in self.pool_after:
                spatial //= 2
        flat_dim = prev_ch * spatial * spatial

        # 分类头：展平 -> Dropout -> 两层全连接 -> 输出层
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
        """Kaiming 初始化（配合 ReLU 让每层输出方差稳定），偏置置零。"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：输入一批图片，输出每个类别的打分。"""
        for layer in self.features:
            x = layer(x)
        x = self.flatten(x)          # (N, 256, 3, 3) -> (N, 2304)
        x = self.drop(x)
        for layer in self.classifier:
            x = layer(x)             # (N, 2304) -> (N, 512)
        return self.out(x)           # (N, 512) -> (N, 10)

    def config(self) -> dict:
        """返回重建这个模型所需的结构参数，供保存 checkpoint 时记录。"""
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
    """ResNet 的基本残差块：两层 3×3 卷积 + 跨层连接。

        输入 x ──┬─→ Conv3x3 → BN → ReLU → Conv3x3 → BN ─→ (+) ─→ ReLU ─→ 输出
                 └────────────── 跨层连接（恒等或 1×1 卷积）─┘

    跨层连接解决的是「深层网络反而更差」的问题。理论上多堆几层不该变差，
    但实际训练中梯度要一层层往回传，层数一多就容易衰减，深层的参数几乎收不到
    有效的梯度。残差连接给梯度开了一条直通道（加法运算的导数恒为 1），
    让梯度可以直接回流到浅层。

    Args:
        in_channels:  输入通道数。
        out_channels: 输出通道数。
        stride:       步长。取 2 时这个块同时负责把边长减半（下采样）。
        residual:     是否使用跨层连接。设为 False 就是一个普通的卷积块，
                      可以直接跑 ResNet 论文里的消融实验（PlainNet）。
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
        # （通道数变了，或者 stride=2 让特征图变小了）
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
    """为 28×28 灰度图改造的 ResNet-18。

    结构（1 个 stem + 4 个 stage，每个 stage 2 个残差块）：

        输入                                    (N, 1, 28, 28)
          Conv2d(1, 64, 3, padding=1) + BN + ReLU (N, 64, 28, 28)   stem
          stage1  2 × BasicBlock(64 -> 64)        (N, 64, 28, 28)
          stage2  2 × BasicBlock(64 -> 128, s=2)  (N, 128, 14, 14)
          stage3  2 × BasicBlock(128 -> 256, s=2) (N, 256, 7, 7)
          stage4  2 × BasicBlock(256 -> 512, s=2) (N, 512, 4, 4)
          AdaptiveAvgPool2d(1)                   (N, 512, 1, 1)
          Flatten                                (N, 512)
          Linear(512, 10)                        (N, 10)   logits

    和原版的两处差别：

    1. **没有开头的 MaxPool**。原版是 7×7 卷积 + 步长 2 的最大池化，直接把
       224 变成 56。28 像素经不起这一刀，所以改成 CIFAR 版的写法：一个 3×3
       的 stem，不做下采样。
    2. **没有全连接大层**。原版最后是 `Linear(512, 1000)`，这里也一样 ——
       因为中间用了全局平均池化，`Linear(512, 10)` 只有 5130 个参数。
       对比 AlexNet 的 289 万参数分类头，这是 ResNet 在结构上的关键改进。

    Args:
        num_blocks: 每个 stage 的残差块数量，默认 (2,2,2,2) 即 ResNet-18。
                    改成 (3,4,6,3) 就是 ResNet-34 的配置。
        widths:     各 stage 的输出通道数，默认 (64,128,256,512)。
        residual:   是否使用残差连接，透传给每个 BasicBlock。
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

        # 保存下来，推理时要用同一套结构重建模型
        self.num_blocks = tuple(num_blocks)
        self.widths = tuple(widths)
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_classes = num_classes
        self.residual = residual

        # stem：3×3 卷积，不做下采样
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.widths[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.widths[0]),
            nn.ReLU(inplace=True),
        )

        # 4 个 stage，stage2 起每个 stage 用 stride=2 把边长减半
        stages: list[nn.Module] = []
        prev_ch = self.widths[0]
        for stage_i, (out_ch, n_blocks) in enumerate(zip(self.widths, self.num_blocks)):
            for block_i in range(n_blocks):
                # 每个 stage 的第一个块负责下采样（第一个 stage 除外）
                stride = 2 if (stage_i > 0 and block_i == 0) else 1
                stages.append(BasicBlock(prev_ch, out_ch, stride=stride, residual=residual))
                prev_ch = out_ch
        self.stages = nn.ModuleList(stages)

        # 全局平均池化 + 单层全连接。池化把 (N, C, H, W) 压成 (N, C, 1, 1)，
        # 于是不管输入多大，送进全连接的都只有 C 维 —— 这是它取代大全连接层的原因。
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(prev_ch, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """卷积层用 Kaiming 初始化，BatchNorm 用默认值（weight=1, bias=0）。"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：输入一批图片，输出每个类别的打分。"""
        x = self.stem(x)
        for block in self.stages:
            x = block(x)
        x = self.pool(x)         # (N, 512, 4, 4) -> (N, 512, 1, 1)
        x = self.flatten(x)      # (N, 512)
        return self.fc(x)        # (N, 512) -> (N, 10)

    def config(self) -> dict:
        """返回重建这个模型所需的结构参数，供保存 checkpoint 时记录。"""
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
    """根据 checkpoint 里存的结构参数重建模型（推理时用）。

    必须按训练时的结构原样重建，否则权重对不上号。

    注意 config 里的 "arch" 字段：Level 1/2 只有一个模型，config 里不需要
    说明是哪个；Level 3 有 AlexNet 和 ResNet 两个，所以要靠它来分发。
    """
    arch = config.get("arch", "alexnet")
    if arch not in ARCHITECTURES:
        raise ValueError(f"未知模型 {arch!r}，可选：{sorted(ARCHITECTURES)}")

    if arch == "alexnet":
        return AlexNet(
            conv_channels=tuple(config["conv_channels"]),
            fc_hidden=tuple(config.get("fc_hidden", (1024, 512))),
            in_channels=config.get("in_channels", 1),
            input_size=config.get("input_size", 28),
            num_classes=config.get("num_classes", 10),
            dropout=config.get("dropout", 0.0),  # 推理时会 eval()，dropout 不生效
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
    # 直接运行本文件时，打印两个模型的结构、参数量和一次前向的形状变化，方便自己核对。
    dummy = torch.randn(4, 1, 28, 28)
    print(f"输入形状: {tuple(dummy.shape)}\n")

    for name, build in (
        ("AlexNet", lambda: AlexNet()),
        ("ResNet18", lambda: ResNet18()),
        ("PlainNet18（无残差连接）", lambda: ResNet18(residual=False)),
    ):
        model = build()
        model.eval()
        print("=" * 62)
        print(f"{name}   可训练参数量: {count_parameters(model):,}")
        print("=" * 62)
        with torch.no_grad():
            out = model(dummy)
        print(f"  输出形状: {tuple(out.shape)}\n")
