"""Multi-layer token models for the active MHDSRA2 architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from .domain import normalize_model_type
from .mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2


def select_mhdsra2_heads(dim: int) -> int:
    """Select a valid MHDSRA2 head count for a hidden dimension.

    中文说明:
    - 调用方 / Called by: `MultiLayerMHDSRA2Model.__init__`
    - 调用对象 / Calls: 内置 `range`, `min`, `max`
    - 作用 / Purpose: 为多层 MHDSRA2 模型选择能整除隐藏维度的保守 head 数
    - 变量 / Variables: `dim` 是隐藏维度, `heads` 是候选头数
    - 接入 / Integration: 新模型构建时复用本函数避免重复 head 选择逻辑
    - 错误处理 / Error handling: 找不到更大可整除值时返回 `1`
    - 关键词 / Keywords:
      mhdsra2|heads|select|dim|divisible|model|multi_layer|attention|config|头数
    """
    for heads in range(min(8, max(1, dim // 16)), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class MultiLayerMHDSRA2Model(nn.Module):
    """Stacked token model backed exclusively by MHDSRA2 layers."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int = 2,
        K: int = 128,
        kr: int = 16,
        chunk_size: int = 256,
        *,
        use_retrieval: bool = False,
        model_type: str = "mhdsra2",
    ) -> None:
        """Create a stacked MHDSRA2 token model.

        中文说明:
        - 调用方 / Called by: `scripts.needle_in_haystack_test.build_niah_model`,
          legacy `MultiLayerDSRAModel`
        - 调用对象 / Calls: `normalize_model_type`, `select_mhdsra2_heads`,
          `MHDSRA2Config`, `MultiHeadDSRA2`, PyTorch layers
        - 作用 / Purpose: 为长上下文 token 任务提供正式多层 MHDSRA2 架构
        - 变量 / Variables:
          `vocab_size/dim/num_layers` 是模型规模, `K/kr/chunk_size` 是记忆和分块配置,
          `use_retrieval` 控制外部召回分支, `model_type` 记录归一化后的架构名
        - 接入 / Integration: 通过 `build_niah_model(model_type="mhdsra2")` 或兼容别名构造
        - 错误处理 / Error handling: 非法架构名、维度或 MHDSRA2 配置会抛出 `ValueError`
        - 关键词 / Keywords:
          mhdsra2|multilayer|model|token|chunked|streaming|slots|retrieval|logits|模型
        """
        super().__init__()
        active_model_type = normalize_model_type(model_type)
        if active_model_type != "mhdsra2":
            raise ValueError(f"Unsupported multi-layer architecture: {model_type}")

        heads = select_mhdsra2_heads(dim)
        self.architecture = active_model_type
        self.dim = dim
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                MultiHeadDSRA2(
                    MHDSRA2Config(
                        dim=dim,
                        heads=heads,
                        slots=K,
                        read_topk=max(1, min(kr, K)),
                        write_topk=max(1, min(kr, K)),
                        local_window=max(1, int(chunk_size)),
                        use_local=True,
                        use_retrieval=use_retrieval,
                        detach_state=True,
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a stacked MHDSRA2 token model over chunked long sequences.

        中文说明:
        - 调用方 / Called by: Needle-In-A-Haystack training/evaluation scripts and tests
        - 调用对象 / Calls: `nn.Embedding`, `nn.LayerNorm`, `MultiHeadDSRA2.forward`, `nn.Linear`
        - 作用 / Purpose: 对 token id 序列执行多层流式 MHDSRA2 前向并返回词表 logits
        - 变量 / Variables:
          `x` 是 `[B,SeqLen]` token ids, `hidden` 是嵌入序列, `chunk` 是当前分块,
          `state_list` 保存每层流式状态, `out_list` 收集每个分块输出
        - 接入 / Integration: 输入 token ids，输出 `[B,SeqLen,vocab_size]` logits
        - 错误处理 / Error handling: 张量维度和底层配置错误由 PyTorch/MHDSRA2 向上抛出
        - 关键词 / Keywords:
          forward|mhdsra2|multilayer|chunked|streaming|state|token|logits|needle|前向
        """
        _, seq_len = x.shape
        hidden = self.embedding(x)
        state_list = [None] * self.num_layers
        out_list = []

        for start in range(0, seq_len, self.chunk_size):
            chunk = hidden[:, start : start + self.chunk_size, :]
            for layer_idx, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                residual = chunk
                chunk_normed = norm(chunk)
                out_chunk, next_state = layer(chunk_normed, state=state_list[layer_idx])
                state_list[layer_idx] = next_state
                chunk = residual + out_chunk
            out_list.append(chunk)

        out = torch.cat(out_list, dim=1)
        out = self.final_norm(out)
        return self.out_proj(out)


class MultiLayerDSRAModel(MultiLayerMHDSRA2Model):
    """Archived DSRA name retained as an MHDSRA2 compatibility alias."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int = 2,
        K: int = 128,
        kr: int = 16,
        chunk_size: int = 256,
        use_orthogonal_update: bool = True,
        use_bypass: bool = True,
        pe_mode: str = "none",
    ) -> None:
        """Create the archived DSRA alias using the active MHDSRA2 model.

        中文说明:
        - 调用方 / Called by: legacy `model_type="dsra"` code paths
        - 调用对象 / Calls: `MultiLayerMHDSRA2Model.__init__`
        - 作用 / Purpose: 将旧 DSRA 多层模型名归档为兼容入口，实际全面使用 MHDSRA2
        - 变量 / Variables:
          `use_orthogonal_update/use_bypass/pe_mode` 是旧参数，仅用于兼容签名；
          `vocab_size/dim/num_layers/K/kr/chunk_size` 传递给 MHDSRA2 架构
        - 接入 / Integration: 外部旧导入无需改名即可获得 MHDSRA2 行为
        - 错误处理 / Error handling: 底层 MHDSRA2 配置错误向上抛出，不吞异常
        - 关键词 / Keywords:
          archived|dsra|alias|mhdsra2|compat|multilayer|model|migration|legacy|归档
        """
        self.archived_dsra_options = {
            "use_orthogonal_update": bool(use_orthogonal_update),
            "use_bypass": bool(use_bypass),
            "pe_mode": pe_mode,
        }
        super().__init__(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            K=K,
            kr=kr,
            chunk_size=chunk_size,
            use_retrieval=False,
            model_type="mhdsra2",
        )
