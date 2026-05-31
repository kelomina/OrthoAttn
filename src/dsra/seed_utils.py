"""Shared deterministic seed helpers for DSRA experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, cudnn_benchmark: bool = False) -> int:
    """Seed Python, NumPy, Torch and CUDA with consistent cuDNN flags.

    中文说明:
    - 调用方 / Called by: experiment scripts such as `scripts.ablation_study`.
    - 调用对象 / Calls: `random.seed`, `np.random.seed`, `torch.manual_seed`,
      `torch.cuda.manual_seed_all`, `torch.use_deterministic_algorithms`.
    - 作用 / Purpose: 统一实验随机性入口，减少脚本之间 seed 设置不一致导致的不可复现。
    - 参数 / Parameters:
      `seed` 是整数随机种子；`cudnn_benchmark` 控制是否允许 cuDNN 自动调优。
    - 返回 / Returns: 规范化后的整数 seed，便于写入报告。
    - 错误处理 / Error handling: PyTorch deterministic 设置不支持时以 warn_only 模式处理。
    - 副作用 / Side effects: 修改全局随机状态与 cuDNN 行为。

    English documentation:
    Function name:
        seed_everything
    Purpose:
        Apply one reproducibility policy across Python, NumPy, Torch and CUDA.
    Called by:
        Experiment scripts.
    Calls:
        `random.seed`, `np.random.seed`, `torch.manual_seed`,
        `torch.cuda.manual_seed_all`, `torch.use_deterministic_algorithms`.
    Parameters:
        - seed: integer seed.
        - cudnn_benchmark: whether cuDNN benchmark mode is enabled.
    Returns:
        The normalized integer seed.
    Error handling:
        Deterministic algorithms are requested with `warn_only=True`.
    Side effects:
        Mutates global RNG states and cuDNN flags.
    English keywords:
        seed, deterministic, reproducibility, torch, numpy, cudnn
    """
    resolved_seed = int(seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = not cudnn_benchmark
        torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.use_deterministic_algorithms(True, warn_only=True)
    return resolved_seed
