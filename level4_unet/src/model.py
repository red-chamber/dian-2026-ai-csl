"""Level 4：U-Net（手写内容擦除）。

结构是原版 U-Net（Ronneberger et al., 2015）的标准形态：编码器逐级下采样、
解码器逐级上采样，同一层级的编码器特征通过跳跃连接直接接到解码器上。

    输入 (1, H, W)
      ├─ enc1: ConvBlock(1   -> 64)   ─────────────────────┐ 跳跃连接
      │   pool                                             │
      ├─ enc2: ConvBlock(64  -> 128)  ──────────────┐      │
      │   pool                                      │      │
      ├─ enc3: ConvBlock(128 -> 256)  ───────┐      │      │
      │   pool                               │      │      │
      ├─ enc4: ConvBlock(256 -> 512)  ─┐     │      │      │
      │   pool                         │     │      │      │
      └─ bottleneck: ConvBlock(512 -> 1024)   │      │      │
          up + concat ──────────────────┘     │      │      │
          dec4: ConvBlock(1024+512 -> 512)    │      │      │
          up + concat ────────────────────────┘      │      │
          dec3: ConvBlock(512+256 -> 256)            │      │
          up + concat ──────────────────────────────┘      │
          dec2: ConvBlock(256+128 -> 128)                   │
          up + concat ─────────────────────────────────────┘
          dec1: ConvBlock(128+64 -> 64)
          输出 Conv1x1(64 -> 1) + Sigmoid

为什么跳跃连接在这里特别重要
------------------------------

编码器每下采样一次，分辨率减半，位置信息就丢一部分。到瓶颈层时特征图只有
原始尺寸的 1/16，语义强但「哪个像素在哪」已经很模糊了。

而这个任务要求的是**逐像素**的修改：手写笔迹覆盖在哪些像素上，就要精确地把
这些像素恢复成底层的印刷内容。纯编码器-解码器靠 1/16 分辨率的特征去还原
原尺寸的细节，边缘一定是糊的 —— 笔迹的边界会留下灰影。

跳跃连接把编码器的高分辨率特征直接送到解码器，等于把「这里原本长什么样」
这个信息原封不动地补给了解码器侧做局部决策。这是它和 Level 2/3 的分类网络
最大的区别：那边的跳跃连接是为了梯度好传，这里是为了**像素级的位置信息**。

输出用 Sigmoid 而不是直接回归：目标图是 [0,1] 的灰度，Sigmoid 把输出约束在
同一区间，避免出现负值或超过 1 的像素。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# 与 Level 1-3 保持一致：脚本直接运行时仓库根不在 sys.path 里，先补上
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.utils import count_parameters  # noqa: E402  (重新导出，保持统一口径)

__all__ = ["UNet", "ConvBlock", "build_model_from_config", "count_parameters"]


class ConvBlock(nn.Module):
    """两层 3×3 卷积，每层后接 BatchNorm 和 ReLU。

    `padding=1` 保证卷积不改变边长，这样解码器和编码器的特征图尺寸能严格对上，
    拼接时不需要裁剪或补齐。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """用于图像到图像恢复的 U-Net。

    Args:
        base_channels: 第一层的通道数，之后每下采样一级翻倍。
                       取 64 时整网约 3100 万参数（原版 U-Net 的量级）。
                       显存紧张时可以降到 32（约 780 万）。
        depth:         下采样次数。4 次意味着输入边长需要能被 16 整除。
        in_channels:   输入通道数。输入图统一转成灰度，所以是 1。
        out_channels:  输出通道数。目标是单通道灰度图，所以是 1。
        bilinear:      上采样方式。False 用转置卷积（原版做法），
                       True 用双线性插值 + 卷积（参数更少、棋盘伪影更少）。
    """

    def __init__(
        self,
        base_channels: int = 64,
        depth: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        bilinear: bool = False,
    ) -> None:
        super().__init__()

        # 保存下来，推理时要用同一套结构重建模型
        self.base_channels = base_channels
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # 编码器：每一级是「ConvBlock + 池化」，通道数逐级翻倍
        channels = [base_channels * (2**i) for i in range(depth)]
        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch))
            prev = ch
        self.pool = nn.MaxPool2d(kernel_size=2)

        # 瓶颈层：再翻一倍
        bottleneck_channels = base_channels * (2**depth)
        self.bottleneck = ConvBlock(prev, bottleneck_channels)

        # 解码器：每一级先上采样，再和对应编码器的特征拼接，最后过一个 ConvBlock。
        # 拼接后通道数是「上采样输出 + 跳跃连接」，所以 ConvBlock 的输入要按这个算。
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = bottleneck_channels
        for ch in reversed(channels):
            if bilinear:
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(ch),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                # 转置卷积：kernel=2、stride=2，正好把边长翻倍
                self.ups.append(nn.ConvTranspose2d(prev, ch, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(ch + ch, ch))  # +ch 是拼接进来的跳跃连接
            prev = ch

        # 输出层：1×1 卷积压到目标通道数，Sigmoid 约束到 [0,1]
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        """卷积用 Kaiming（配合 ReLU 稳定方差），BatchNorm 用默认的 weight=1/bias=0。"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：输入带手写图，输出擦除后的图。

        Args:
            x: 形状 (N, 1, H, W)，H、W 需要能被 2**depth 整除。
        Returns:
            形状 (N, 1, H, W)，取值在 [0,1]。
        """
        # 编码：把每一级的特征存下来，解码时要用
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # 解码：上采样 -> 拼接跳跃连接 -> ConvBlock。
        # skips 是正序存的，这里要倒着取才能和层级对上。
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            # 尺寸本该严格相等；若输入边长不是 2**depth 的整数倍，这里会差 1 像素，
            # 统一在推理脚本里按倍数补齐解决，模型内部不做事后裁剪。
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError(
                    f"上采样后尺寸 {tuple(x.shape[-2:])} 与跳跃连接 {tuple(skip.shape[-2:])} 不一致，"
                    f"请把输入边长补齐到 2**{self.depth}={2 ** self.depth} 的整数倍"
                )
            x = decoder(torch.cat([skip, x], dim=1))

        return self.activation(self.head(x))

    def config(self) -> dict:
        """返回重建这个模型所需的结构参数，供保存 checkpoint 时记录。"""
        return {
            "arch": "unet",
            "base_channels": self.base_channels,
            "depth": self.depth,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "bilinear": self.bilinear,
        }


def build_model_from_config(config: dict) -> UNet:
    """根据 checkpoint 里存的结构参数重建模型（推理时用）。

    必须按训练时的结构原样重建，否则权重对不上号。
    """
    return UNet(
        base_channels=config.get("base_channels", 64),
        depth=config.get("depth", 4),
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 1),
        bilinear=config.get("bilinear", False),
    )


if __name__ == "__main__":
    # 直接运行本文件时，打印结构、参数量和一次前向的形状变化，方便自己核对。
    crop = 256
    dummy = torch.randn(2, 1, crop, crop)
    print(f"输入形状: {tuple(dummy.shape)}\n")

    for base in (32, 64):
        model = UNet(base_channels=base).eval()
        with torch.no_grad():
            out = model(dummy)
        print(f"base_channels={base:<3} 参数量 {count_parameters(model):>11,}  "
              f"输出 {tuple(out.shape)}")

    model = UNet(base_channels=64).eval()
    print("\n各阶段形状变化（base_channels=64）:")
    with torch.no_grad():
        x = dummy[:1]
        print(f"  输入          {tuple(x.shape)}")
        skips = []
        for i, encoder in enumerate(model.encoders, start=1):
            x = encoder(x)
            skips.append(x)
            print(f"  enc{i}          {tuple(x.shape)}")
            x = model.pool(x)
        x = model.bottleneck(x)
        print(f"  bottleneck    {tuple(x.shape)}")
        for i, (up, decoder, skip) in enumerate(
            zip(model.ups, model.decoders, reversed(skips)), start=1
        ):
            x = up(x)
            print(f"  up{i}           {tuple(x.shape)}")
            x = decoder(torch.cat([skip, x], dim=1))
            print(f"  dec{i}         {tuple(x.shape)}")
        out = model.activation(model.head(x))
        print(f"  输出          {tuple(out.shape)}   取值 [{out.min():.3f}, {out.max():.3f}]")
