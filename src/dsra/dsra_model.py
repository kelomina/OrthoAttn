"""Multi-layer token models for the active MHDSRA2 architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from .domain import normalize_model_type, select_mhdsra2_heads as _domain_select_mhdsra2_heads
from .infrastructure import PagedMemoryRepository
from .mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2


def select_mhdsra2_heads(dim: int) -> int:
    """Compatibility wrapper for the shared MHDSRA2 head-count policy.

    中文说明:
    - 调用方 / Called by: `MultiLayerMHDSRA2Model.__init__`
    - 调用对象 / Calls: `domain.select_mhdsra2_heads`
    - 作用 / Purpose: 保留旧导入位置，同时复用领域层统一 head 选择规则
    - 变量 / Variables: `dim` 是隐藏维度, `heads` 是候选头数
    - 接入 / Integration: 新模型构建时复用本函数，外部旧导入无需迁移
    - 错误处理 / Error handling: 非法维度由领域层 helper 抛出 `ValueError`
    - 关键词 / Keywords:
      mhdsra2|heads|select|dim|divisible|model|multi_layer|attention|config|头数
    """
    return _domain_select_mhdsra2_heads(dim)


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
        mhdsra2_config_override: dict | None = None,
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
        - Note:
           detach_state=True is the memory-safe default (gradient is truncated
           across chunk boundaries). Set to False via mhdsra2_config_override
           for shorter sequences where full BPTT gradient flow is affordable.
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
        base_cfg = MHDSRA2Config(
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
        if mhdsra2_config_override:
            for key, value in mhdsra2_config_override.items():
                if hasattr(base_cfg, key):
                    setattr(base_cfg, key, value)
            base_cfg.__post_init__()
        self.layers = nn.ModuleList(
            [MultiHeadDSRA2(base_cfg) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def update_momentum(self) -> None:
        """Update slow Momentum-QKV projections for every MHDSRA2 layer.

        中文说明:
        - 调用方 / Called by: training loops after `optimizer.step()` when `momentum_qkv=True`
        - 调用对象 / Calls: `MultiHeadDSRA2.update_momentum`
        - 作用 / Purpose: 提供模型级便利入口，避免训练脚本手动逐层遍历更新 slow QKV
        - 参数 / Parameters: 无
        - 返回 / Returns: None
        - 错误处理 / Error handling: 单层更新异常直接向上抛出，不吞掉训练错误
        - 副作用 / Side effects: 原地更新启用 Momentum-QKV 层的 slow projection 权重
        - 关键词 / Keywords:
          momentum|qkv|ema|update|multilayer|mhdsra2|training|optimizer|动量

        English documentation:
        Function name:
            update_momentum
        Purpose:
            Provide one model-level call that forwards EMA QKV updates to each
            MHDSRA2 layer after an optimizer step.
        """
        for layer in self.layers:
            layer.update_momentum()

    def _new_retrieval_repositories(self) -> list[PagedMemoryRepository]:
        """Create per-layer paged retrieval memories for one independent forward pass.

        中文说明:
        - 调用方 / Called by: `forward`, `forward_selected_logits`
        - 调用对象 / Calls: `PagedMemoryRepository`
        - 作用 / Purpose: 为每层创建独立 CPU 分页 K/V 记忆，使 `use_retrieval=True`
          的多层 token 模型在 chunk 之间真正使用 retrieval 分支。
        - 返回 / Returns: 与 `self.layers` 一一对应的仓储列表；禁用 retrieval 的层返回禁用仓储。
        - 错误处理 / Error handling: 仓储初始化错误直接向上抛出。
        - 副作用 / Side effects: 只创建本次 forward 内部使用的新仓储，避免独立样本串记忆。

        English documentation:
        Function name:
            _new_retrieval_repositories
        Purpose:
            Build one paged K/V memory repository per MHDSRA2 layer for a single
            independent forward call.
        Called by:
            `forward` and `forward_selected_logits`.
        Calls:
            `PagedMemoryRepository`.
        Returns:
            A repository list aligned with `self.layers`.
        Side effects:
            Allocates fresh CPU-side retrieval memories for the current call only.
        English keywords:
            retrieval, paged memory, multilayer, mhdsra2, forward, chunk
        """
        return [
            PagedMemoryRepository(
                enabled=bool(layer.cfg.use_retrieval),
                dtype=torch.float32,
                query_pooling=layer.cfg.retrieval_query_pooling,
            )
            for layer in self.layers
        ]

    def _prepare_layer_retrieval(
        self,
        layer: MultiHeadDSRA2,
        repository: PagedMemoryRepository,
        chunk_normed: torch.Tensor,
        state,
        sequence_lengths: torch.Tensor | None = None,
        return_metadata: bool = False,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, object] | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Retrieve old K/V and prepare current K/V for later append.

        中文说明:
        - 调用方 / Called by: `forward`, `forward_selected_logits`
        - 调用对象 / Calls: `MultiHeadDSRA2.qkv`, `_to_heads`,
          `PagedMemoryRepository.retrieve`
        - 作用 / Purpose: 在当前 chunk 前向前，先用 query 从历史分页记忆中召回 K/V；
          同时计算当前 chunk 的 key/value heads，等待前向成功后写入仓储。
        - 参数 / Parameters: `layer` 是当前 MHDSRA2 层，`repository` 是对应层的分页记忆，
          `chunk_normed` 是 LayerNorm 后输入，`state` 提供当前流式位置。
        - 返回 / Returns:
          `(retrieved_k, retrieved_v, retrieved_mask, retrieval_metadata, key_heads, value_heads)`。
        - 错误处理 / Error handling: 禁用 retrieval 时返回四个 `None`；仓储错误直接抛出。
        - 副作用 / Side effects: 本函数只读取仓储，不写入；写入由调用方在前向成功后执行。

        English documentation:
        Function name:
            _prepare_layer_retrieval
        Purpose:
            Fetch previous paged K/V for the current chunk and precompute current
            K/V heads for append after a successful forward pass.
        Called by:
            `forward` and `forward_selected_logits`.
        Calls:
            `layer.qkv`, `layer._to_heads`, and `PagedMemoryRepository.retrieve`.
        Returns:
            `(retrieved_k, retrieved_v, retrieved_mask, retrieval_metadata, key_heads, value_heads)`.
        Side effects:
            None; appending is intentionally delayed until the caller finishes
            the current layer forward.
        English keywords:
            retrieval, key, value, chunk, no self recall, paged memory
        """
        if not layer.cfg.use_retrieval:
            return None, None, None, None, None, None

        query, key, value = layer.qkv(chunk_normed).chunk(3, dim=-1)
        query_heads = layer._to_heads(query)
        key_heads = layer._to_heads(key)
        value_heads = layer._to_heads(value)
        state_position = 0 if state is None else state.position
        max_position = state_position
        if sequence_lengths is not None:
            current_position = torch.full_like(sequence_lengths, int(state_position))
            max_position = torch.minimum(sequence_lengths, current_position)
        if return_metadata:
            retrieved_k, retrieved_v, retrieved_mask, retrieval_metadata = repository.retrieve(
                query_heads.detach(),
                chunk_normed.device,
                max_position=max_position,
                return_mask=True,
                return_metadata=True,
            )
        else:
            retrieved_k, retrieved_v, retrieved_mask = repository.retrieve(
                query_heads.detach(),
                chunk_normed.device,
                max_position=max_position,
                return_mask=True,
            )
            retrieval_metadata = None
        return retrieved_k, retrieved_v, retrieved_mask, retrieval_metadata, key_heads, value_heads

    def _normalize_sequence_lengths(
        self,
        sequence_lengths: int | torch.Tensor | None,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor | None:
        """Normalize optional true sequence lengths for padded batch retrieval.

        中文说明:
        - 调用方 / Called by: `forward`, `forward_selected_logits`
        - 调用对象 / Calls: `torch.as_tensor`, tensor validation operations
        - 作用 / Purpose: 将变长 batch 的真实长度规范为 `[B]` CPU tensor，供外部检索逐样本裁剪
        - 参数 / Parameters: `sequence_lengths` 是每个样本真实 token 数；None 表示所有样本等长
        - 返回 / Returns: None 或 `[B]` CPU long tensor
        - 错误处理 / Error handling: 长度数不匹配、非正或超过当前 `seq_len` 时抛 `ValueError`
        - 副作用 / Side effects: 无。
        """
        if sequence_lengths is None:
            return None
        if isinstance(sequence_lengths, int):
            normalized = torch.full((batch_size,), sequence_lengths, dtype=torch.long)
        else:
            normalized = torch.as_tensor(sequence_lengths, dtype=torch.long).detach().cpu().flatten()
            if normalized.numel() == 1:
                normalized = normalized.expand(batch_size).clone()
        if normalized.numel() != batch_size:
            raise ValueError(
                f"expected one sequence length per batch item, got {normalized.numel()} "
                f"lengths for batch_size={batch_size}"
            )
        if bool(((normalized <= 0) | (normalized > seq_len)).any().item()):
            raise ValueError(f"sequence lengths must be within [1, {seq_len}]")
        return normalized

    def _normalize_selected_positions(
        self,
        positions: int | torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Normalize per-batch token positions for memory-bounded logit selection.

        中文说明:
        - 调用方 / Called by: `forward_selected_logits`
        - 调用对象 / Calls: `torch.as_tensor`, `torch.full`, tensor shape/range checks
        - 作用 / Purpose: 将单个位置或逐 batch 位置标准化为 CPU long tensor，支持负索引
        - 参数 / Parameters:
          `positions` 是单个全局 token 位置或 `[B]` 逐样本位置；`batch_size/seq_len`
          来自输入 token 序列，必须为正整数
        - 返回 / Returns: `[B]` CPU long tensor；非法位置抛出 `ValueError`
        - 接入 / Integration: 仅由模型层内部调用，不涉及事务、文件、网络或外部服务
        - 错误处理 / Error handling: batch 数不匹配或位置越界时直接抛出 `ValueError`
        - 副作用 / Side effects: 无；只创建小型 CPU tensor
        - 并发与幂等 / Concurrency and idempotency: 纯函数式转换，可重复调用
        - 关键词 / Keywords:
          position|selected_logits|niah|memory|streaming|batch|negative_index|mhdsra2|token|位置

        English documentation:
        Function name:
            _normalize_selected_positions
        Purpose:
            Normalize scalar or per-batch selected token positions for streaming logits.
        Called by:
            `forward_selected_logits`.
        Calls:
            `torch.as_tensor`, `torch.full`, and tensor validation operations.
        Parameters:
            - positions: int or tensor, selected global token positions.
            - batch_size: int, number of input samples.
            - seq_len: int, input sequence length.
        Returns:
            CPU long tensor with one valid position per batch item.
        Error handling:
            Raises `ValueError` for unsupported shapes or out-of-range positions.
        Side effects:
            None.
        Transaction boundary:
            Not applicable.
        Concurrency and idempotency:
            Reentrant and idempotent for the same input.
        English keywords:
            position, selected_logits, niah, memory, streaming, batch, negative_index, mhdsra2, token, validation
        """
        if isinstance(positions, int):
            normalized = torch.full((batch_size,), positions, dtype=torch.long)
        else:
            normalized = torch.as_tensor(positions, dtype=torch.long).detach().cpu().flatten()
            if normalized.numel() == 1:
                normalized = normalized.expand(batch_size).clone()

        if normalized.numel() != batch_size:
            raise ValueError(
                f"expected one selected position per batch item, got {normalized.numel()} "
                f"positions for batch_size={batch_size}"
            )

        normalized = torch.where(normalized < 0, normalized + seq_len, normalized)
        if bool(((normalized < 0) | (normalized >= seq_len)).any().item()):
            raise ValueError(f"selected positions must be within [0, {seq_len})")
        return normalized

    def forward_selected_logits(
        self,
        x: torch.Tensor,
        positions: int | torch.Tensor,
        stage_id: int | None = None,
        context_id: int | None = None,
        *,
        sequence_lengths: int | torch.Tensor | None = None,
        return_hidden: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, object]]:
        """Run MHDSRA2 over a long sequence and return logits only at selected positions.

        When return_hidden=True, also returns the pre-out_proj normalized hidden
        states of shape [B, dim] for diagnostic callers that need selected-token
        representations without materializing full-sequence logits. When
        return_aux=True, also returns lightweight MHDSRA2 diagnostics collected
        from the final layer's latest processed chunk.
        """
        if x.dim() != 2:
            raise ValueError(f"expected token ids with shape [B, SeqLen], got {tuple(x.shape)}")

        batch_size, seq_len = x.shape
        selected_positions = self._normalize_selected_positions(positions, batch_size, seq_len)
        normalized_lengths = self._normalize_sequence_lengths(sequence_lengths, batch_size, seq_len)
        if normalized_lengths is not None and bool(
            (selected_positions >= normalized_lengths).any().item()
        ):
            raise ValueError("selected positions must be before each sample's sequence length")
        model_device = self.embedding.weight.device
        state_list = [None] * self.num_layers
        retrieval_repositories = self._new_retrieval_repositories()
        logits_by_batch: list[torch.Tensor | None] = [None] * batch_size
        hidden_by_batch: list[torch.Tensor | None] = [None] * batch_size if return_hidden else None
        latest_aux_by_layer: list[dict[str, object] | None] = [None] * self.num_layers
        selected_aux_by_layer: list[dict[str, object] | None] = [None] * self.num_layers

        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            token_chunk = x[:, start:end].to(device=model_device, non_blocking=True)
            chunk = self.embedding(token_chunk)

            for layer_idx, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                residual = chunk
                chunk_normed = norm(chunk)
                (
                    retrieved_k,
                    retrieved_v,
                    retrieved_mask,
                    retrieval_metadata,
                    key_heads,
                    value_heads,
                ) = self._prepare_layer_retrieval(
                    layer,
                    retrieval_repositories[layer_idx],
                    chunk_normed,
                    state_list[layer_idx],
                    normalized_lengths,
                    return_metadata=return_aux,
                )
                layer_result = layer(
                    chunk_normed,
                    state=state_list[layer_idx],
                    retrieved_k=retrieved_k,
                    retrieved_v=retrieved_v,
                    retrieved_mask=retrieved_mask,
                    return_aux=return_aux,
                    stage_id=stage_id,
                    context_id=context_id,
                )
                if return_aux:
                    out_chunk, next_state, aux = layer_result
                    if retrieval_metadata is not None:
                        aux["retrieval_metadata"] = retrieval_metadata
                    latest_aux_by_layer[layer_idx] = aux
                else:
                    out_chunk, next_state = layer_result
                state_list[layer_idx] = next_state
                if key_heads is not None and value_heads is not None:
                    retrieval_repositories[layer_idx].append(key_heads, value_heads)
                chunk = residual + out_chunk

            selected_mask = (selected_positions >= start) & (selected_positions < end)
            if bool(selected_mask.any().item()):
                batch_idx_cpu = selected_mask.nonzero(as_tuple=True)[0]
                local_idx_cpu = selected_positions[batch_idx_cpu] - start
                batch_idx = batch_idx_cpu.to(device=model_device)
                local_idx = local_idx_cpu.to(device=model_device)
                selected_hidden = chunk[batch_idx, local_idx, :]
                hidden_normed = self.final_norm(selected_hidden)
                selected_logits = self.out_proj(hidden_normed)
                for row_idx, batch_id in enumerate(batch_idx_cpu.tolist()):
                    logits_by_batch[batch_id] = selected_logits[row_idx : row_idx + 1]
                    if return_hidden:
                        hidden_by_batch[batch_id] = hidden_normed[row_idx : row_idx + 1]
                if return_aux:
                    selected_aux_by_layer = []
                    for aux in latest_aux_by_layer:
                        if not isinstance(aux, dict):
                            selected_aux_by_layer.append(aux)
                            continue
                        selected_aux = dict(aux)
                        selected_aux["selected_batch_indices"] = batch_idx.detach().cpu()
                        gate_by_token = aux.get("gate_retrieval_by_token")
                        if isinstance(gate_by_token, torch.Tensor) and gate_by_token.dim() == 2:
                            selected_aux["selected_gate_retrieval_by_sample"] = gate_by_token[
                                batch_idx,
                                local_idx,
                            ]
                        gate_by_token_for_loss = aux.get("gate_retrieval_by_token_for_loss")
                        if (
                            isinstance(gate_by_token_for_loss, torch.Tensor)
                            and gate_by_token_for_loss.dim() == 2
                        ):
                            selected_aux[
                                "selected_gate_retrieval_by_sample_for_loss"
                            ] = gate_by_token_for_loss[batch_idx, local_idx]
                        weights_by_token = aux.get("retrieval_token_weight_by_token")
                        if (
                            isinstance(weights_by_token, torch.Tensor)
                            and weights_by_token.dim() == 3
                        ):
                            selected_aux[
                                "selected_retrieval_token_weight_by_sample"
                            ] = weights_by_token[batch_idx, local_idx, :]
                        weights_by_token_for_loss = aux.get(
                            "retrieval_token_weight_by_token_for_loss"
                        )
                        if (
                            isinstance(weights_by_token_for_loss, torch.Tensor)
                            and weights_by_token_for_loss.dim() == 3
                        ):
                            selected_aux[
                                "selected_retrieval_token_weight_by_sample_for_loss"
                            ] = weights_by_token_for_loss[batch_idx, local_idx, :]
                        retrieval_metadata = aux.get("retrieval_metadata")
                        if isinstance(retrieval_metadata, dict):
                            selected_metadata: dict[str, object] = {}
                            positions = retrieval_metadata.get("positions")
                            if isinstance(positions, torch.Tensor):
                                if positions.dim() == 1:
                                    selected_metadata["positions"] = positions
                                elif positions.dim() >= 2:
                                    selected_metadata["positions"] = positions[batch_idx, ...]
                            mask = retrieval_metadata.get("mask")
                            if isinstance(mask, torch.Tensor):
                                if mask.dim() == 1:
                                    selected_metadata["mask"] = mask
                                elif mask.dim() >= 2:
                                    selected_metadata["mask"] = mask[batch_idx, ...]
                            counts = retrieval_metadata.get("retrieved_token_counts")
                            if isinstance(counts, torch.Tensor):
                                selected_metadata["retrieved_token_counts"] = counts[
                                    batch_idx
                                ]
                            if "max_position" in retrieval_metadata:
                                selected_metadata["max_position"] = retrieval_metadata[
                                    "max_position"
                                ]
                            if selected_metadata:
                                selected_aux["selected_retrieval_metadata"] = selected_metadata
                        selected_aux_by_layer.append(selected_aux)

        if any(item is None for item in logits_by_batch):
            raise RuntimeError("failed to collect logits for all selected positions")
        logits = torch.cat([item for item in logits_by_batch if item is not None], dim=0)
        if return_aux:
            aux_payload: dict[str, object] = {
                "layers": selected_aux_by_layer,
                "last_layer": selected_aux_by_layer[-1] if selected_aux_by_layer else None,
                "latest_layers": latest_aux_by_layer,
                "latest_last_layer": latest_aux_by_layer[-1] if latest_aux_by_layer else None,
            }
            if return_hidden:
                hidden = torch.cat([item for item in hidden_by_batch if item is not None], dim=0)
                return logits, hidden, aux_payload
            return logits, aux_payload
        if return_hidden:
            hidden = torch.cat([item for item in hidden_by_batch if item is not None], dim=0)
            return logits, hidden
        return logits

    def forward(
        self,
        x: torch.Tensor,
        stage_id: int | None = None,
        context_id: int | None = None,
        *,
        sequence_lengths: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a stacked MHDSRA2 token model over chunked long sequences.

        中文说明:
        - 调用方 / Called by: full-sequence compatibility tests and legacy evaluation scripts
        - 调用对象 / Calls: `nn.Embedding`, `nn.LayerNorm`, `MultiHeadDSRA2.forward`, `nn.Linear`
        - 作用 / Purpose: 对 token id 序列执行多层流式 MHDSRA2 前向并返回全序列词表 logits
        - 变量 / Variables:
          `x` 是 `[B,SeqLen]` token ids, `token_chunk` 是当前分块 token ids,
          `state_list` 保存每层流式状态, `out_list` 收集每个分块输出
        - 接入 / Integration: 输入 token ids，输出 `[B,SeqLen,vocab_size]` logits；只监督少量位置时优先用
          `forward_selected_logits` 降低显存
        - 错误处理 / Error handling: 张量维度和底层配置错误由 PyTorch/MHDSRA2 向上抛出
        - 关键词 / Keywords:
          forward|mhdsra2|multilayer|chunked|streaming|state|token|logits|compat|前向
        """
        batch_size, seq_len = x.shape
        normalized_lengths = self._normalize_sequence_lengths(sequence_lengths, batch_size, seq_len)
        model_device = self.embedding.weight.device
        state_list = [None] * self.num_layers
        retrieval_repositories = self._new_retrieval_repositories()
        out_list = []

        for start in range(0, seq_len, self.chunk_size):
            token_chunk = x[:, start : start + self.chunk_size].to(
                device=model_device, non_blocking=True
            )
            chunk = self.embedding(token_chunk)
            for layer_idx, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                residual = chunk
                chunk_normed = norm(chunk)
                retrieved_k, retrieved_v, retrieved_mask, _, key_heads, value_heads = self._prepare_layer_retrieval(
                    layer,
                    retrieval_repositories[layer_idx],
                    chunk_normed,
                    state_list[layer_idx],
                    normalized_lengths,
                )
                out_chunk, next_state = layer(
                    chunk_normed,
                    state=state_list[layer_idx],
                    retrieved_k=retrieved_k,
                    retrieved_v=retrieved_v,
                    retrieved_mask=retrieved_mask,
                    stage_id=stage_id,
                    context_id=context_id,
                )
                state_list[layer_idx] = next_state
                if key_heads is not None and value_heads is not None:
                    retrieval_repositories[layer_idx].append(key_heads, value_heads)
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
