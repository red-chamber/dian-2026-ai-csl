'''模型的各模块声明
    MLP 类：主要模块定义
    count_parameters：统计参数量
    build_model_from_config：根据 checkpoint 里存的结构参数重建模型
'''

import torch
import torch.nn as nn

class MLP(nn.Module):

    # def 函数名(参数列表) -> 返回类型注解: 函数体
    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (512, 256), # 各隐藏层的宽度。tuple 元组不可变，避免了默认值初始化一次的 bug
        in_features: int = 28 * 28, # 输入维度
        num_classes: int = 10, # 输出类别数
        dropout: float = 0.2, # Dropout 概率，训练时随机把一部分神经元置零，抑制过拟合
    ) -> None:
        super().__init__()

        # 相当于声明
        self.hidden_sizes = tuple(hidden_sizes) # tuple 再次防止传列表进来
        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout_p = dropout

        self.flatten = nn.Flatten() # 创建了一个可调用对象，类似函数，用于展平 1 到最后一维

        # 输入层(784) -> 隐藏层1(512) -> 隐藏层2(256) -> 输出层(10)
        # 一个隐藏层 = Linear -> ReLU -> Dropout
        layers = [] 
        prev_dim = in_features                              # 前一层输出的维度，也是这一层输入的维度
        for hidden_dim in self.hidden_sizes:                # 当前层的输出维度
            layers.append(nn.Linear(prev_dim, hidden_dim))  # 线性变换(仿射)：y = xW^T + b
            layers.append(nn.ReLU(inplace=True))            # ReLU：负数变 0，正数不变
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))        # 训练时随机失活
            prev_dim = hidden_dim                           # 更新

        self.hidden_layers = nn.ModuleList(layers)          # 把子模块纳入父模块(MLP)的注册表，使 PyTorch 能识别
        self.classifier = nn.Linear(prev_dim, num_classes)  # 分类器，只有 Linear

        self._init_weights()

    # 初始化权重
    def _init_weights(self) -> None:

        # 遍历每个模块
        for module in self.modules():
            # 过滤出 Linear 层
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu") # Kaiming 初始化
                nn.init.zeros_(module.bias) # 把偏置全置零

    # 前向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.flatten(x)                  # (N, 1, 28, 28) -> (N, 784)
        for layer in self.hidden_layers:     # (N, 784) -> (N, 512) -> (N, 256)
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
    """统计参数量
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
    """根据 checkpoint 里存的结构参数重建模型（推理时用）

    必须按训练时的结构原样重建，否则权重对不上号。
    """
    return MLP(
        hidden_sizes=tuple(config["hidden_sizes"]),
        in_features=config.get("in_features", 28 * 28),
        num_classes=config.get("num_classes", 10),
        dropout=config.get("dropout", 0.0),  # 推理时会 eval()，dropout 不生效
    )