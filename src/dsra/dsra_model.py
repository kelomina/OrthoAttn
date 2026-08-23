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
        use_retrieval_span_predictor: bool = False,
        retrieval_span_structure_features: bool = False,
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
          `use_retrieval` 控制外部召回分支, `use_retrieval_span_predictor` 只在显式
          NIAH 实验中创建候选 span 打分头, `retrieval_span_structure_features`
          默认关闭，仅让显式结构实验组把 pair/source 结构特征送入 span predictor,
          `model_type` 记录归一化后的架构名
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
        self.use_retrieval_span_predictor = bool(use_retrieval_span_predictor)
        self.retrieval_span_structure_features = bool(retrieval_span_structure_features)
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
        if self.use_retrieval_span_predictor:
            span_feature_count = 8 if self.retrieval_span_structure_features else 4
            self.retrieval_span_predictor = nn.Sequential(
                nn.Linear(dim * 4 + span_feature_count, dim),
                nn.GELU(),
                nn.Linear(dim, 1),
            )
        else:
            self.retrieval_span_predictor = None

    def score_retrieval_span_candidates(
        self,
        hidden_query: torch.Tensor,
        candidate_token_ids: torch.Tensor,
        source_token_ids: torch.Tensor,
        candidate_positions: torch.Tensor,
        query_positions: torch.Tensor,
        *,
        candidate_mask: torch.Tensor | None = None,
        candidate_weights: torch.Tensor | None = None,
        candidate_scores: torch.Tensor | None = None,
        candidate_pair_mask: torch.Tensor | None = None,
        source_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score retrieved candidates for the opt-in NIAH evidence span readout.

        中文说明:
        - 调用方 / Called by: `scripts.needle_in_haystack_test` 的默认关闭
          retrieval span predictor 实验。
        - 调用对象 / Calls: `embedding`, `retrieval_span_predictor`。
        - 作用 / Purpose: 只在显式开启 `use_retrieval_span_predictor=True` 时，
          根据 query hidden、候选 token、候选可复制来源 token、相对位置和检索分数
          给每个 retrieval 候选打分。默认模型不会创建这个头，也不会改变 logits。
        - 参数 / Parameters:
          `hidden_query` 是 `[B,D]` query 表征；`candidate_token_ids/source_token_ids`
          是 `[B,R]` token id；`candidate_positions/query_positions` 用于提供相对位置特征；
          `candidate_mask` 标记有效候选，`candidate_weights/scores` 是可选检索诊断信号；
          `candidate_pair_mask/source_positions` 只在显式结构特征实验组中提供。
        - 返回 / Returns: `[B,R]` span logits；无效候选会被置为很小值。
        - 错误处理 / Error handling: 未启用 predictor 或形状不匹配时抛 `ValueError/RuntimeError`。
        - 副作用 / Side effects: 无；不写检索记忆、不修改模型状态。
        - 关键词 / Keywords:
          retrieval|span_predictor|niah|candidate|readout|evidence|mhdsra2|证据

        English documentation:
        Function name:
            score_retrieval_span_candidates
        Purpose:
            Score selected retrieval candidates for an explicit, opt-in evidence
            span decoder without changing the default MHDSRA2 forward path.
        """
        if self.retrieval_span_predictor is None:
            raise RuntimeError("retrieval span predictor is not enabled")
        if hidden_query.dim() != 2:
            raise ValueError(f"hidden_query must be [B,D], got {tuple(hidden_query.shape)}")
        if candidate_token_ids.shape != source_token_ids.shape:
            raise ValueError("candidate_token_ids and source_token_ids must have matching shape")
        if candidate_token_ids.dim() != 2:
            raise ValueError(
                f"candidate token ids must be [B,R], got {tuple(candidate_token_ids.shape)}"
            )
        if candidate_positions.shape != candidate_token_ids.shape:
            raise ValueError("candidate_positions must match candidate_token_ids shape")
        batch_size, candidate_count = candidate_token_ids.shape
        if hidden_query.shape[0] != batch_size:
            raise ValueError("hidden_query batch size must match candidate batch size")
        query_positions = torch.as_tensor(
            query_positions,
            dtype=torch.long,
            device=hidden_query.device,
        ).flatten()
        if query_positions.numel() == 1 and batch_size > 1:
            query_positions = query_positions.expand(batch_size)
        if query_positions.numel() != batch_size:
            raise ValueError("query_positions must provide one position per batch row")

        device = hidden_query.device
        safe_candidate_ids = candidate_token_ids.to(device=device, dtype=torch.long).clamp(
            0,
            self.embedding.num_embeddings - 1,
        )
        safe_source_ids = source_token_ids.to(device=device, dtype=torch.long).clamp(
            0,
            self.embedding.num_embeddings - 1,
        )
        query = hidden_query.to(dtype=torch.float32).unsqueeze(1).expand(
            -1,
            candidate_count,
            -1,
        )
        candidate_embed = self.embedding(safe_candidate_ids).to(dtype=torch.float32)
        source_embed = self.embedding(safe_source_ids).to(dtype=torch.float32)
        interaction = query * source_embed

        positions = candidate_positions.to(device=device, dtype=torch.float32)
        query_pos = query_positions.to(device=device, dtype=torch.float32).view(batch_size, 1)
        relative_distance = ((query_pos - positions).clamp_min(0.0) / query_pos.clamp_min(1.0))
        if candidate_weights is None:
            weight_feature = torch.zeros(batch_size, candidate_count, device=device)
        else:
            weight_feature = candidate_weights.to(device=device, dtype=torch.float32)
        if candidate_scores is None:
            score_feature = torch.zeros(batch_size, candidate_count, device=device)
        else:
            score_feature = candidate_scores.to(device=device, dtype=torch.float32)
            finite_scores = torch.isfinite(score_feature)
            score_feature = torch.where(finite_scores, score_feature, torch.zeros_like(score_feature))
            score_feature = score_feature.clamp(-50.0, 50.0)
        available_feature = (
            torch.ones(batch_size, candidate_count, device=device)
            if candidate_mask is None
            else candidate_mask.to(device=device, dtype=torch.float32)
        )
        position_feature = torch.stack(
            [relative_distance, weight_feature, score_feature, available_feature],
            dim=-1,
        )
        if self.retrieval_span_structure_features:
            if candidate_pair_mask is None:
                pair_feature = torch.zeros(batch_size, candidate_count, device=device)
            else:
                pair_feature = candidate_pair_mask.to(device=device, dtype=torch.float32)
            if source_positions is None:
                source_delta = torch.zeros(batch_size, candidate_count, device=device)
            else:
                source_pos = source_positions.to(device=device, dtype=torch.float32)
                source_delta = ((source_pos - positions) / query_pos.clamp_min(1.0)).clamp(
                    -1.0,
                    1.0,
                )
            source_is_right = (source_delta > 0.0).to(dtype=torch.float32)
            source_is_self = (source_delta == 0.0).to(dtype=torch.float32)
            structure_feature = torch.stack(
                [pair_feature, source_delta, source_is_right, source_is_self],
                dim=-1,
            )
            position_feature = torch.cat([position_feature, structure_feature], dim=-1)
        features = torch.cat(
            [query, candidate_embed, source_embed, interaction, position_feature],
            dim=-1,
        )
        logits = self.retrieval_span_predictor(features).squeeze(-1)
        if candidate_mask is not None:
            logits = logits.masked_fill(
                ~candidate_mask.to(device=device, dtype=torch.bool),
                torch.finfo(logits.dtype).min,
            )
        return logits

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
                max_tokens=layer.cfg.retrieval_max_tokens,
                query_pooling=layer.cfg.retrieval_query_pooling,
                neighbor_span=layer.cfg.retrieval_neighbor_span,
                neighbor_direction=layer.cfg.retrieval_neighbor_direction,
                neighbor_seed_multiplier=layer.cfg.retrieval_neighbor_seed_multiplier,
                neighbor_budget_mode=layer.cfg.retrieval_neighbor_budget_mode,
                page_score_mode=layer.cfg.page_score_mode,
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
        train_retrieval_evidence_positions: int | torch.Tensor | None = None,
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
          `train_retrieval_evidence_positions` 只用于训练期合成任务的显式证据候选注入，
          默认 None，因此不会改变 baseline、validation 或 test 的检索行为。

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
        if train_retrieval_evidence_positions is not None:
            (
                evidence_k,
                evidence_v,
                _evidence_positions,
                evidence_mask,
                evidence_metadata,
            ) = repository.retrieve_positions(
                train_retrieval_evidence_positions,
                chunk_normed.device,
                max_position=max_position,
                dtype=query_heads.dtype,
                return_mask=True,
                return_metadata=True,
            )
            (
                retrieved_k,
                retrieved_v,
                retrieved_mask,
                retrieval_metadata,
            ) = self._merge_train_retrieval_evidence_candidates(
                retrieved_k,
                retrieved_v,
                retrieved_mask,
                retrieval_metadata,
                evidence_k,
                evidence_v,
                evidence_mask,
                evidence_metadata,
                batch_size=int(chunk_normed.shape[0]),
                device=chunk_normed.device,
            )
        return retrieved_k, retrieved_v, retrieved_mask, retrieval_metadata, key_heads, value_heads

    def _merge_train_retrieval_evidence_candidates(
        self,
        retrieved_k: torch.Tensor | None,
        retrieved_v: torch.Tensor | None,
        retrieved_mask: torch.Tensor | None,
        retrieval_metadata: dict[str, object] | None,
        evidence_k: torch.Tensor | None,
        evidence_v: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
        evidence_metadata: dict[str, object] | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, object] | None,
    ]:
        """Merge normal retrieval candidates with explicit train-only evidence.

        中文说明:
        - 调用方 / Called by: `_prepare_layer_retrieval`
        - 调用对象 / Calls: `torch.cat`
        - 作用 / Purpose: 在 evidence supervision 显式开启时，把 gold evidence 位置的 K/V
          追加到普通分页检索候选后面，让辅助 loss 能监督“已拿到证据时是否使用证据”。
        - 参数 / Parameters: `retrieved_*` 是普通检索候选，`evidence_*` 是按训练证据位置查出的候选。
        - 返回 / Returns: 合并后的 K/V、mask、metadata；没有 evidence 命中时保持普通检索结果。
        - 错误处理 / Error handling: K/V rank 不兼容时抛 `ValueError`，避免静默拼错 batch。
        - 副作用 / Side effects: 无；不写仓储、不修改输入字典。
        - 关键词 / Keywords:
          evidence|candidate_injection|retrieval|selected_aux|train_only|mhdsra2|证据注入

        English documentation:
        Function name:
            _merge_train_retrieval_evidence_candidates
        Purpose:
            Append explicit train-only evidence candidates to normal paged
            retrieval results while preserving the legacy default path.
        """
        if evidence_k is None or evidence_v is None or evidence_mask is None:
            return retrieved_k, retrieved_v, retrieved_mask, retrieval_metadata
        if evidence_mask.numel() == 0 or not bool(evidence_mask.any().item()):
            return retrieved_k, retrieved_v, retrieved_mask, retrieval_metadata
        if retrieved_k is None or retrieved_v is None:
            evidence_only_metadata = self._merge_retrieval_metadata_for_train_evidence(
                None,
                evidence_metadata,
                batch_size=batch_size,
                device=device,
            )
            return evidence_k, evidence_v, evidence_mask, evidence_only_metadata
        if retrieved_k.dim() != evidence_k.dim() or retrieved_v.dim() != evidence_v.dim():
            raise ValueError("normal retrieval and evidence retrieval ranks must match")
        if retrieved_k.dim() != 4:
            raise ValueError("train evidence candidate injection expects [B,H,R,d] retrieval")

        merged_k = torch.cat([retrieved_k, evidence_k.to(device=retrieved_k.device)], dim=2)
        merged_v = torch.cat([retrieved_v, evidence_v.to(device=retrieved_v.device)], dim=2)

        normal_mask = self._normalize_retrieval_mask_for_merge(
            retrieved_mask,
            batch_size=batch_size,
            width=int(retrieved_k.shape[2]),
            device=device,
        )
        gold_mask = self._normalize_retrieval_mask_for_merge(
            evidence_mask,
            batch_size=batch_size,
            width=int(evidence_k.shape[2]),
            device=device,
        )
        merged_mask = torch.cat([normal_mask, gold_mask], dim=1)
        if batch_size == 1:
            merged_mask = merged_mask.view(-1)

        merged_metadata = self._merge_retrieval_metadata_for_train_evidence(
            retrieval_metadata,
            evidence_metadata,
            batch_size=batch_size,
            device=device,
        )
        return merged_k, merged_v, merged_mask, merged_metadata

    def _normalize_retrieval_mask_for_merge(
        self,
        mask: torch.Tensor | None,
        *,
        batch_size: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize retrieval validity mask to `[B,R]` before candidate merge."""
        if mask is None:
            return torch.ones(batch_size, width, device=device, dtype=torch.bool)
        normalized = mask.to(device=device, dtype=torch.bool)
        if normalized.dim() == 1:
            if batch_size != 1:
                raise ValueError("1-D retrieval mask is only valid for batch_size=1")
            normalized = normalized.view(1, -1)
        if normalized.dim() != 2:
            raise ValueError("retrieval mask must be [R] or [B,R]")
        if normalized.shape != (batch_size, width):
            raise ValueError(
                f"retrieval mask shape {tuple(normalized.shape)} does not match "
                f"batch_size={batch_size}, width={width}"
            )
        return normalized

    def _metadata_positions_and_mask_for_merge(
        self,
        metadata: dict[str, object] | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Extract metadata positions/mask as `[B,R]` tensors for train evidence merge."""
        if not isinstance(metadata, dict):
            return None, None
        positions = metadata.get("positions")
        mask = metadata.get("mask")
        if not isinstance(positions, torch.Tensor) or not isinstance(mask, torch.Tensor):
            return None, None
        pos_tensor = positions.to(device=device, dtype=torch.long)
        mask_tensor = mask.to(device=device, dtype=torch.bool)
        if pos_tensor.dim() == 1:
            if batch_size != 1:
                raise ValueError("1-D metadata positions are only valid for batch_size=1")
            pos_tensor = pos_tensor.view(1, -1)
            mask_tensor = mask_tensor.view(1, -1)
        if pos_tensor.dim() != 2 or mask_tensor.dim() != 2:
            raise ValueError("retrieval metadata positions/mask must be [R] or [B,R]")
        if pos_tensor.shape != mask_tensor.shape:
            raise ValueError("retrieval metadata positions and mask shapes must match")
        if pos_tensor.shape[0] != batch_size:
            raise ValueError("retrieval metadata batch size does not match input batch")
        return pos_tensor, mask_tensor

    def _merge_retrieval_metadata_for_train_evidence(
        self,
        retrieval_metadata: dict[str, object] | None,
        evidence_metadata: dict[str, object] | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, object] | None:
        """Merge normal and train-evidence retrieval metadata for selected aux."""
        evidence_positions, evidence_mask = self._metadata_positions_and_mask_for_merge(
            evidence_metadata,
            batch_size=batch_size,
            device=device,
        )
        if evidence_positions is None or evidence_mask is None:
            return retrieval_metadata
        normal_positions, normal_mask = self._metadata_positions_and_mask_for_merge(
            retrieval_metadata,
            batch_size=batch_size,
            device=device,
        )
        if normal_positions is None or normal_mask is None:
            merged_positions = evidence_positions
            merged_mask = evidence_mask
        else:
            merged_positions = torch.cat([normal_positions, evidence_positions], dim=1)
            merged_mask = torch.cat([normal_mask, evidence_mask], dim=1)
        counts = merged_mask.to(dtype=torch.long).sum(dim=1)
        if batch_size == 1:
            merged_positions_out: torch.Tensor = merged_positions.view(-1)
            merged_mask_out: torch.Tensor = merged_mask.view(-1)
        else:
            merged_positions_out = merged_positions
            merged_mask_out = merged_mask
        merged: dict[str, object] = {
            "positions": merged_positions_out,
            "mask": merged_mask_out,
            "retrieved_token_counts": counts.to(device=device),
            "train_evidence_injected": True,
        }
        if isinstance(retrieval_metadata, dict) and "max_position" in retrieval_metadata:
            merged["max_position"] = retrieval_metadata["max_position"]
        elif isinstance(evidence_metadata, dict) and "max_position" in evidence_metadata:
            merged["max_position"] = evidence_metadata["max_position"]
        if isinstance(evidence_metadata, dict) and "lookup_positions" in evidence_metadata:
            merged["train_evidence_lookup_positions"] = evidence_metadata["lookup_positions"]
        return merged

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

    def _normalize_train_retrieval_evidence_positions(
        self,
        positions: int | torch.Tensor | None,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor | None:
        """Normalize optional train-only evidence positions to `[B]`.

        中文说明:
        - 调用方 / Called by: `forward_selected_logits`
        - 调用对象 / Calls: `torch.as_tensor`, tensor validation operations
        - 作用 / Purpose: 把 NIAH 训练期传入的 gold evidence 位置标准化为 CPU tensor；
          `-1` 表示该样本不注入证据，其余位置必须在当前序列范围内。默认单证据
          口径返回 `[B]`；显式 span predictor 实验可传 `[B,N]` 同时注入 key/value。
        - 参数 / Parameters: `positions` 是 None、标量、`[B]` 或 `[B,N]` 位置；`batch_size/seq_len`
          来自当前 token batch。
        - 返回 / Returns: None 或 `[B]` CPU long tensor。
        - 错误处理 / Error handling: 形状不匹配或位置越界时抛 `ValueError`。
        - 副作用 / Side effects: 无。
        - 关键词 / Keywords:
          train_only|evidence|retrieval|position|batch|selected_logits|mhdsra2|证据

        English documentation:
        Function name:
            _normalize_train_retrieval_evidence_positions
        Purpose:
            Validate opt-in train-only evidence coordinates before selected
            retrieval candidate injection.
        """
        if positions is None:
            return None
        if isinstance(positions, int):
            normalized = torch.full((batch_size,), positions, dtype=torch.long)
        else:
            raw = torch.as_tensor(positions, dtype=torch.long).detach().cpu()
            if raw.dim() == 0:
                normalized = raw.view(1).expand(batch_size).clone()
            elif raw.dim() == 1:
                normalized = raw
                if normalized.numel() == 1:
                    normalized = normalized.expand(batch_size).clone()
                if normalized.numel() != batch_size:
                    raise ValueError(
                        f"expected one train evidence position per batch item, got "
                        f"{normalized.numel()} positions for batch_size={batch_size}"
                    )
            elif raw.dim() == 2:
                if raw.shape[0] != batch_size:
                    raise ValueError(
                        f"expected train evidence positions for {batch_size} samples, "
                        f"got {raw.shape[0]}"
                    )
                normalized = raw.clone()
            else:
                raise ValueError("train evidence positions must be scalar, [B], or [B,N]")
        if normalized.dim() == 1 and normalized.numel() != batch_size:
            raise ValueError(
                f"expected one train evidence position per batch item, got {normalized.numel()} "
                f"positions for batch_size={batch_size}"
            )
        valid_or_disabled = normalized >= 0
        if bool(((normalized[valid_or_disabled] >= seq_len)).any().item()):
            raise ValueError(f"train evidence positions must be within [0, {seq_len}) or -1")
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
        return_retrieval_projection_aux: bool = False,
        train_retrieval_evidence_positions: int | torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, object]]:
        """Run MHDSRA2 over a long sequence and return logits only at selected positions.

        When return_hidden=True, also returns the pre-out_proj normalized hidden
        states of shape [B, dim] for diagnostic callers that need selected-token
        representations without materializing full-sequence logits. When
        return_aux=True, also returns lightweight MHDSRA2 diagnostics collected
        from the final layer's latest processed chunk.

        中文说明:
        `train_retrieval_evidence_positions` 是默认关闭的训练期辅助入口。NIAH 这类合成任务
        知道正确答案来自哪个历史 token；当 evidence loss 显式启用时，调用方可以传入这些
        位置，让包含 query 的 chunk 在 retrieval 候选中看到该证据。validation/test 不应传入
        该参数，避免把评估变成“提前给答案位置”的开卷测试。
        """
        if x.dim() != 2:
            raise ValueError(f"expected token ids with shape [B, SeqLen], got {tuple(x.shape)}")

        batch_size, seq_len = x.shape
        selected_positions = self._normalize_selected_positions(positions, batch_size, seq_len)
        normalized_train_evidence_positions = self._normalize_train_retrieval_evidence_positions(
            train_retrieval_evidence_positions,
            batch_size,
            seq_len,
        )
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
            selected_mask_for_chunk = (selected_positions >= start) & (selected_positions < end)
            evidence_positions_for_chunk = None
            if normalized_train_evidence_positions is not None and bool(
                selected_mask_for_chunk.any().item()
            ):
                if normalized_train_evidence_positions.dim() == 1:
                    evidence_positions_for_chunk = torch.where(
                        selected_mask_for_chunk,
                        normalized_train_evidence_positions,
                        torch.full_like(normalized_train_evidence_positions, -1),
                    )
                else:
                    evidence_positions_for_chunk = torch.where(
                        selected_mask_for_chunk.view(-1, 1),
                        normalized_train_evidence_positions,
                        torch.full_like(normalized_train_evidence_positions, -1),
                    )

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
                    train_retrieval_evidence_positions=evidence_positions_for_chunk,
                )
                layer_result = layer(
                    chunk_normed,
                    state=state_list[layer_idx],
                    retrieved_k=retrieved_k,
                    retrieved_v=retrieved_v,
                    retrieved_mask=retrieved_mask,
                    return_aux=return_aux,
                    return_projection_aux=return_retrieval_projection_aux,
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

            if bool(selected_mask_for_chunk.any().item()):
                batch_idx_cpu = selected_mask_for_chunk.nonzero(as_tuple=True)[0]
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
                        scores_by_token = aux.get("retrieval_token_score_by_token")
                        if (
                            isinstance(scores_by_token, torch.Tensor)
                            and scores_by_token.dim() == 3
                        ):
                            selected_aux[
                                "selected_retrieval_token_score_by_sample"
                            ] = scores_by_token[batch_idx, local_idx, :]
                        scores_by_token_for_loss = aux.get(
                            "retrieval_token_score_by_token_for_loss"
                        )
                        if (
                            isinstance(scores_by_token_for_loss, torch.Tensor)
                            and scores_by_token_for_loss.dim() == 3
                        ):
                            selected_aux[
                                "selected_retrieval_token_score_by_sample_for_loss"
                            ] = scores_by_token_for_loss[batch_idx, local_idx, :]
                        query_projection_by_token = aux.get("retrieval_query_projection_by_token")
                        if (
                            isinstance(query_projection_by_token, torch.Tensor)
                            and query_projection_by_token.dim() == 4
                        ):
                            selected_aux[
                                "selected_retrieval_query_projection"
                            ] = query_projection_by_token[
                                batch_idx,
                                :,
                                local_idx,
                                :,
                            ]
                        query_projection_by_token_for_loss = aux.get(
                            "retrieval_query_projection_by_token_for_loss"
                        )
                        if (
                            isinstance(query_projection_by_token_for_loss, torch.Tensor)
                            and query_projection_by_token_for_loss.dim() == 4
                        ):
                            selected_aux[
                                "selected_retrieval_query_projection_for_loss"
                            ] = query_projection_by_token_for_loss[
                                batch_idx,
                                :,
                                local_idx,
                                :,
                            ]
                        key_projection_by_sample = aux.get(
                            "retrieval_key_projection_by_sample"
                        )
                        if (
                            isinstance(key_projection_by_sample, torch.Tensor)
                            and key_projection_by_sample.dim() == 4
                        ):
                            selected_aux[
                                "selected_retrieval_key_projection"
                            ] = key_projection_by_sample[batch_idx, :, :, :]
                        key_projection_by_sample_for_loss = aux.get(
                            "retrieval_key_projection_by_sample_for_loss"
                        )
                        if (
                            isinstance(key_projection_by_sample_for_loss, torch.Tensor)
                            and key_projection_by_sample_for_loss.dim() == 4
                        ):
                            selected_aux[
                                "selected_retrieval_key_projection_for_loss"
                            ] = key_projection_by_sample_for_loss[batch_idx, :, :, :]
                        key_projection_by_token = aux.get("retrieval_key_projection_by_token")
                        if (
                            isinstance(key_projection_by_token, torch.Tensor)
                            and key_projection_by_token.dim() == 5
                        ):
                            selected_aux[
                                "selected_retrieval_key_projection"
                            ] = key_projection_by_token[
                                batch_idx,
                                :,
                                local_idx,
                                :,
                                :,
                            ]
                        key_projection_by_token_for_loss = aux.get(
                            "retrieval_key_projection_by_token_for_loss"
                        )
                        if (
                            isinstance(key_projection_by_token_for_loss, torch.Tensor)
                            and key_projection_by_token_for_loss.dim() == 5
                        ):
                            selected_aux[
                                "selected_retrieval_key_projection_for_loss"
                            ] = key_projection_by_token_for_loss[
                                batch_idx,
                                :,
                                local_idx,
                                :,
                                :,
                            ]
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
                            if "train_evidence_injected" in retrieval_metadata:
                                selected_metadata["train_evidence_injected"] = retrieval_metadata[
                                    "train_evidence_injected"
                                ]
                            lookup_positions = retrieval_metadata.get(
                                "train_evidence_lookup_positions"
                            )
                            if isinstance(lookup_positions, torch.Tensor):
                                if lookup_positions.dim() >= 1 and lookup_positions.shape[0] == batch_size:
                                    selected_metadata[
                                        "train_evidence_lookup_positions"
                                    ] = lookup_positions[batch_idx, ...]
                                else:
                                    selected_metadata[
                                        "train_evidence_lookup_positions"
                                    ] = lookup_positions
                            for list_field in (
                                "selected_page_ranges_by_sample",
                                "page_candidate_positions_by_sample",
                                "top_token_positions_by_sample",
                                "seed_token_positions_by_sample",
                            ):
                                list_value = retrieval_metadata.get(list_field)
                                if (
                                    isinstance(list_value, list)
                                    and len(list_value) == batch_size
                                ):
                                    selected_metadata[list_field] = [
                                        list_value[int(idx)]
                                        for idx in batch_idx.detach().cpu().tolist()
                                    ]
                            for tensor_field in (
                                "selected_page_ranges",
                                "page_candidate_positions",
                                "top_token_positions",
                                "seed_token_positions",
                            ):
                                tensor_value = retrieval_metadata.get(tensor_field)
                                if isinstance(tensor_value, torch.Tensor):
                                    selected_metadata[tensor_field] = tensor_value
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
