"""Level 1-3 共用的训练基础设施。

为什么要有这个包
----------------

Level 1 和 Level 2 各带了一份完整的 train.py / infer.py，两份几乎逐行相同；
Level 3 再复制一份就是第三份。这个包把三处共用的东西抽出来：

    utils.py       路径常量、随机种子、参数量统计、指标组装与导出
    data.py        数据集注册表（MNIST / Fashion-MNIST）与 DataLoader 构建
    engine.py      训练循环、评估、完整的训练流程、收敛曲线
    checkpoint.py  权重的存与读（按 model_config 重建模型）
    plots.py       通用绘图（收敛曲线）

各 Level 的 src/ 里只保留真正不同的部分：模型定义、命令行接口、以及
各自特有的分析脚本（比如 Level 2 的四角度对比）。

使用方式
--------

各 Level 的入口脚本（train.py / infer.py / compare.py）在 import 本包之前，
需要先把仓库根目录放进 sys.path —— 因为脚本是用 `python level1_mlp/src/train.py`
这种方式直接运行的，Python 只会把脚本自己所在目录加进 sys.path，仓库根不在里面。
每个入口脚本开头都有这几行：

    import sys
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parents[N]     # 往上数到仓库根
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

本包内部模块之间用相对 import（`from .utils import ...`），所以只要仓库根进了
sys.path，`import common.xxx` 就能正常工作。
"""
