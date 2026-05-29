"""Experiment configuration for DSRA language model PPL evaluation.

中文说明：
本模块定义 LMExperimentConfig，作为所有语言模型 PPL 评测试验的统一配置。
"""

from __future__ import annotations

from pathlib import Path

from torch import device as torch_device


# ---- 默认路径 ----
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_WIKITEXT2_DIR = str(_PROJECT_ROOT / "data" / "wikitext-2")
_DEFAULT_WIKITEXT103_DIR = str(_PROJECT_ROOT / "data" / "wikitext-103")


class LMExperimentConfig:
    """冻结论配置类，包含 PPL 评测试验的所有超参数。

    所有字段通过 __init__ 参数设置，实例化后不可修改。
    """

    # ----- 数据集 -----
    dataset: str  # "wikitext2" 或 "wikitext103"
    data_dir: str  # 数据集存放目录
    max_chars: int | None  # None = 使用完整数据集

    # ----- 数据加载 -----
    seq_len: int  # 序列长度
    batch_size: int  # 训练 batch size
    eval_batch_size: int  # 评测 batch size（可大于训练 batch）

    # ----- 训练 -----
    max_steps: int  # 最大训练步数
    lr: float  # 学习率
    warmup_steps: int  # warmup 步数
    eval_interval: int  # valid 集评测间隔（步数）
    clip_grad_norm: float  # 梯度裁剪范数

    # ----- 模型（公共） -----
    dim: int  # 隐藏层维度
    heads: int  # 注意力头数
    num_layers: int  # 层数
    model_type: str  # "standard" 或 "mhdsra2"

    # ----- 模型（MHDSRA2 专用） -----
    slots: int  # 槽位数
    chunk_size: int  # 分块大小
    local_window_mult: int  # local window = chunk_size * local_window_mult

    # ----- 设备与随机性 -----
    device: str  # "auto"、"cuda" 或 "cpu"
    seed: int  # 随机种子

    def __init__(
        self,
        *,
        dataset: str = "wikitext103",
        data_dir: str | None = None,
        max_chars: int | None = None,
        seq_len: int = 512,
        batch_size: int = 8,
        eval_batch_size: int | None = None,
        max_steps: int = 200000,
        lr: float = 3e-4,
        warmup_steps: int = 1000,
        eval_interval: int = 5000,
        clip_grad_norm: float = 1.0,
        dim: int = 512,
        heads: int = 4,
        num_layers: int = 6,
        model_type: str = "standard",
        slots: int = 128,
        chunk_size: int = 128,
        local_window_mult: int = 4,
        device: str = "auto",
        seed: int = 42,
    ):
        self.dataset = dataset
        # data_dir 默认值：根据 dataset 自动选择
        if data_dir is None:
            if dataset == "wikitext103":
                self.data_dir = _DEFAULT_WIKITEXT103_DIR
            else:
                self.data_dir = _DEFAULT_WIKITEXT2_DIR
        else:
            self.data_dir = data_dir
        self.max_chars = max_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size * 2
        self.max_steps = max_steps
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.eval_interval = eval_interval
        self.clip_grad_norm = clip_grad_norm
        self.dim = dim
        self.heads = heads
        self.num_layers = num_layers
        self.model_type = model_type
        self.slots = slots
        self.chunk_size = chunk_size
        self.local_window_mult = local_window_mult
        self.device = device
        self.seed = seed

    def resolve_torch_device(self) -> torch_device:
        """将 device 字符串转换为 torch.device。

        中文说明:
        - 调用方 / Called by: 训练脚本或评测脚本在构建模型前调用
        - 调用对象 / Calls: `torch.cuda.is_available`
        - 作用 / Purpose: 解析 "auto" 为当前环境可用设备，统一设备选择逻辑
        - 接入 / Integration: 在创建模型或张量前调用，返回 `torch.device`
        - 错误处理 / Error handling: 非法 device 字符串会由 `torch.device` 抛出异常
        - 关键词 / Keywords: device, torch, cuda, cpu, auto, 设备解析
        """
        if self.device == "auto":
            import torch
            return torch_device("cuda" if torch.cuda.is_available() else "cpu")
        return torch_device(self.device)

    def to_dict(self) -> dict:
        """转换为普通字典，兼容旧版 LMConfig 风格。

        中文说明:
        - 调用方 / Called by: 旧版训练循环或序列化逻辑
        - 调用对象 / Calls: 无外部函数，仅构造 `dict`
        - 作用 / Purpose: 将配置字段投影为扁平字典，供尚未迁移至 config 对象的旧代码使用
        - 接入 / Integration: 在需要 `dict` 格式超参数的位置调用
        - 关键词 / Keywords: to_dict, dict, serialize, compatibility, 兼容字典
        """
        return {
            "dim": self.dim,
            "heads": self.heads,
            "num_layers": self.num_layers,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "max_steps": self.max_steps,
            "eval_interval": self.eval_interval,
            "warmup_steps": self.warmup_steps,
            "device": self.device,
            "data_dir": self.data_dir,
            "dataset": self.dataset,
        }

    def variant_label(self) -> str:
        """返回变体标签，用于报告文件名。

        中文说明:
        - 调用方 / Called by: 报告生成工具 `report_utils`
        - 调用对象 / Calls: 无外部函数，仅构造 `str`
        - 作用 / Purpose: 根据模型类型生成唯一标签，用于区分不同实验的报告文件
        - 接入 / Integration: 在生成报告文件路径时调用
        - 关键词 / Keywords: variant, label, report, filename, 变体标签
        """
        parts = [self.model_type, f"d{self.dim}", f"l{self.num_layers}"]
        if self.model_type == "mhdsra2":
            parts.append(f"s{self.slots}")
            parts.append(f"c{self.chunk_size}")
        return "_".join(parts)
