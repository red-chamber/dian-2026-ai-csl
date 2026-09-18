import torch
import torch.nn as nn


class MLP(nn.Module):
    """用于 MNIST 分类的多层感知机。

    Args:
        hidden_sizes: 各隐藏层的宽度。默认 (512, 256) 表示两层隐藏层。
        in_features:  输入维度，MNIST 展平后是 28*28=784。
        num_classes:  输出类别数，MNIST 是 10（数字 0~9）。
        dropout:      Dropout 概率，训练时随机把一部分神经元置零，用于抑制过拟合。
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (512, 256),  # def  函数名(参数列表)->返回类型注解: 函数体
        in_features: int = 28 * 28,
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # 保存下来，推理时要用同一套结构重建模型
        self.hidden_sizes = tuple(hidden_sizes)
        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout_p = dropout

        # Flatten 是「数据流动」的第一步：把 (N, 1, 28, 28) 变成 (N, 784)
        #   N     = batch size，一次喂进来多少张图
        #   1     = 通道数，MNIST 是灰度图所以只有 1 个通道
        #   28,28 = 高和宽
        self.flatten = nn.Flatten()

        # 逐层堆叠「线性层 + 激活函数」。
        # 这里用 nn.ModuleList 而不是 nn.Sequential，是为了在 forward 里
        # 一步一步写清楚数据是怎么流的，而不是藏成一行黑盒。
        layers: list[nn.Module] = []
        prev_dim = in_features
        for hidden_dim in self.hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))  # 仿射变换：y = xW^T + b
            layers.append(nn.ReLU(inplace=True))            # 非线性：负数变 0，正数不变
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))        # 训练时随机失活
            prev_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)
        # 输出层：最后一个隐藏层 -> 10 个类别打分
        self.classifier = nn.Linear(prev_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """权重初始化。

        全连接层的默认初始化对 MNIST 这种小任务够用，但显式写出来更清楚：
        权重用 Kaiming 初始化（配合 ReLU 能让每层输出方差保持稳定），偏置置零。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：输入一批图片，输出每个类别的打分。

        Args:
            x: 形状 (N, 1, 28, 28) 或已展平的 (N, 784)。
        Returns:
            形状 (N, 10) 的打分（logits）。
        """
        x = self.flatten(x)                  # (N, 1, 28, 28) -> (N, 784)
        for layer in self.hidden_layers:     # (N, 784) -> ... -> (N, 256)
            x = layer(x)
        logits = self.classifier(x)          # (N, 256) -> (N, 10)
        return logits

    def config(self) -> dict:
        """返回重建这个模型所需的结构参数，供保存 checkpoint 时记录。"""
        return {
            "hidden_sizes": list(self.hidden_sizes),
            "in_features": self.in_features,
            "num_classes": self.num_classes,
            "dropout": self.dropout_p,
        }


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """统计参数量。

    Level 2 要从「参数量」角度对比 MLP 与 CNN，所以这个函数放在 model.py 里，
    两个模型共用同一个口径。

    MLP 的参数量手算方式（不靠 PyTorch 也能核对）：
        第一层  784*512 + 512    = 401920
        第二层  512*256 + 256    = 131328
        输出层  256*10  + 10     =   2570
        合计                     = 535818
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def build_model_from_config(config: dict) -> MLP:
    """根据 checkpoint 里存的结构参数重建模型（推理时用）。

    必须按训练时的结构原样重建，否则权重对不上号。
    """
    return MLP(
        hidden_sizes=tuple(config["hidden_sizes"]),
        in_features=config.get("in_features", 28 * 28),
        num_classes=config.get("num_classes", 10),
        dropout=config.get("dropout", 0.0),  # 推理时会 eval()，dropout 不生效
    )


if __name__ == "__main__":
    # 直接运行本文件时，打印结构、参数量和一次前向的形状变化，方便自己核对。
    model = MLP()
    print(model)
    print(f"\n可训练参数量: {count_parameters(model):,}")

    dummy = torch.randn(4, 1, 28, 28)
    print(f"\n输入形状 : {tuple(dummy.shape)}")
    flat = model.flatten(dummy)
    print(f"展平后   : {tuple(flat.shape)}   <- 28*28 = {28 * 28}")
    with torch.no_grad():
        out = model(dummy)
    print(f"输出形状 : {tuple(out.shape)}   <- 每张图 10 个类别的打分")
