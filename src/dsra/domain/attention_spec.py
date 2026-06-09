"""Domain specification for streaming attention layers."""

from __future__ import annotations

from dataclasses import dataclass


def select_mhdsra2_heads(dim: int) -> int:
    """Select a conservative MHDSRA2 head count for a hidden dimension.

    中文说明:
    - 调用方 / Called by: `DSRA_Chunk_Layer.__init__`, `MultiLayerMHDSRA2Model.__init__`
    - 调用对象 / Calls: 内置 `range`, `min`, `max`
    - 作用 / Purpose: 集中选择能整除隐藏维度的多头数量，避免兼容层和模型层各维护一份规则
    - 参数 / Parameters: `dim` 是隐藏维度，必须为正整数
    - 返回 / Returns: 最大不超过 8 且能整除 `dim` 的保守 head 数，找不到时返回 1
    - 错误处理 / Error handling: 非正维度直接抛出 `ValueError`
    - 关键词 / Keywords:
      mhdsra2|heads|select|dim|divisible|domain|attention|config|helper|头数

    English documentation:
    Function name:
        select_mhdsra2_heads
    Purpose:
        Share the MHDSRA2 head-selection policy across compatibility and model
        constructors without coupling those modules to each other.
    """
    dim = int(dim)
    if dim < 1:
        raise ValueError("dim must be positive")
    for heads in range(min(8, max(1, dim // 16)), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


@dataclass(frozen=True)
class AttentionLayerSpec:
    """Describe a DSRA-compatible attention layer without binding infrastructure.

    中文说明:
    - 调用方 / Called by: `DSRA_Chunk_Layer.__init__`
    - 调用对象 / Calls: 内置 `dataclass` 校验流程与 `ValueError`
    - 作用 / Purpose: 作为领域层配置对象，统一表达维度、槽位、top-k、窗口与位置编码模式
    - 变量 / Variables:
      `dim` 是隐藏维度, `slots` 是全局记忆槽数量, `read_topk/write_topk` 是稀疏路由宽度,
      `local_window` 是局部 KV 窗口, `pe_mode` 是兼容旧入口的位置模式
    - 接入 / Integration: 应用层或模型层构造注意力实现前先创建本规格对象
    - 错误处理 / Error handling: 非法维度、槽位、top-k、窗口或位置模式会抛出 `ValueError`
    - 关键词 / Keywords:
      attention|domain|spec|dsra|mhdsra2|slots|topk|window|position|领域规格
    """

    dim: int
    slots: int
    read_topk: int
    write_topk: int
    local_window: int
    pe_mode: str = "none"

    def __post_init__(self) -> None:
        """Validate immutable attention layer specification.

        中文说明:
        - 调用方 / Called by: `dataclasses` 在 `AttentionLayerSpec` 初始化后自动调用
        - 调用对象 / Calls: 无外部函数，仅使用比较表达式与集合成员检查
        - 作用 / Purpose: 保证领域配置进入应用层前已经满足基本约束
        - 变量 / Variables:
          `allowed_pe_modes` 是允许的位置编码模式集合，其余字段来自当前规格对象
        - 接入 / Integration: 新增配置字段时在此补充领域级约束
        - 错误处理 / Error handling: 检测到不合法字段时立即抛出 `ValueError`
        - 关键词 / Keywords:
          validate|post_init|domain|config|pe_mode|dim|slots|topk|window|校验
        """
        allowed_pe_modes = {"none", "rope", "alibi", "timestamps"}
        if self.dim < 1:
            raise ValueError("dim must be positive")
        if self.slots < 1:
            raise ValueError("slots must be positive")
        if self.read_topk < 1 or self.write_topk < 1:
            raise ValueError("top-k values must be positive")
        if self.local_window < 0:
            raise ValueError("local_window must be non-negative")
        if self.pe_mode not in allowed_pe_modes:
            raise ValueError(f"Unsupported pe_mode: {self.pe_mode}")
