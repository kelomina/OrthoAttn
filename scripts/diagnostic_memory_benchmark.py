"""Runnable A/B/C long-context diagnostics wired for the existing benchmark flow.

The diagnostics are implemented as streaming synthetic probes so the benchmark can
exercise 2M-token style settings without materializing the whole sequence in
memory. Each probe reuses the existing DSRA, MHDSRA2, sliding-window attention,
and linear-attention layers with deterministic identity-style initialization.
"""
from __future__ import annotations

import argparse
import gc
import math
import random
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_mhdsra2_vs_dsra import build_benchmark_comparison_row
from scripts.toy_task_associative_recall import (
    LinearAttentionChunkLayer,
    SlidingWindowAttentionChunkLayer,
)
from src.dsra.dsra_layer import DSRA_Chunk_Layer
from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2
from src.dsra.mhdsra2.paged_exact_memory import PagedExactMemory

MODEL_ORDER = (
    "dsra",
    "mhdsra2_without_paged_recall",
    "mhdsra2_with_paged_recall",
    "sliding_window_attention",
    "linear_attention",
)
MODEL_LABELS = {
    "dsra": "Original DSRA",
    "mhdsra2_without_paged_recall": "MH-DSRA-v2 (no paged recall)",
    "mhdsra2_with_paged_recall": "MH-DSRA-v2 (paged recall)",
    "sliding_window_attention": "Sliding window attention",
    "linear_attention": "Linear attention",
}
PRIMARY_METRIC_BY_SUITE = {
    "diagnostic_a_exact_recall": "exact_match_rate",
    "diagnostic_b_error_override": "latest_fact_accuracy",
    "diagnostic_c_anti_fixation": "counterexample_accuracy",
}
LOWER_IS_BETTER_METRICS = {"stale_fact_rate", "majority_trap_rate"}


@dataclass(frozen=True)
class DiagnosticCase:
    """One synthetic streaming case for a diagnostic suite.

    中文说明:
    - 调用方 / Called by: `build_exact_recall_cases`, `build_error_override_cases`,
      `build_anti_fixation_cases`, `run_diagnostic_suite`
    - 调用对象 / Calls: 无
    - 作用 / Purpose: 封装单个诊断样例的流式序列规格、目标值与展示元数据
    - 变量 / Variables:
      `suite` 诊断套件名, `case_id` 样例标识, `seq_len` 总长度, `query_position`
      查询位置, `special_tokens` 稀疏覆盖 token 向量映射, `target_value_id`
      目标值编号, `candidate_value_ids` 候选值列表, `metadata` 额外说明
    - 接入 / Integration: 新增诊断子任务时直接构造并追加 `DiagnosticCase`
    - 错误处理 / Error handling: 依赖上游构造保证 `query_position < seq_len`
      且 `special_tokens` 中张量维度一致
    - 关键词 / Keywords:
      diagnostic_case|suite|streaming|synthetic|query|target|metadata|sparse_tokens|benchmark|诊断样例
    """

    suite: str
    case_id: str
    seq_len: int
    query_position: int
    dim: int
    target_value_id: int
    candidate_value_ids: tuple[int, ...]
    special_tokens: dict[int, torch.Tensor]
    metadata: dict


def _set_identity_linear(linear: nn.Linear) -> None:
    """Overwrite a linear layer with an identity matrix.

    中文说明:
    - 调用方 / Called by: `_build_dsra_layer`, `_build_mhdsra2_layer`,
      `_build_sliding_window_layer`, `_build_linear_attention_layer`
    - 调用对象 / Calls: `torch.eye`, `torch.no_grad`
    - 作用 / Purpose: 把现有投影层改造成确定性的 identity 形式，方便诊断实验复用
    - 变量 / Variables: `linear` 为待覆盖线性层，要求输入输出维度一致
    - 接入 / Integration: 其他合成 probe 若需要去除随机初始化，也可直接复用
    - 错误处理 / Error handling: 若层形状不是方阵则抛出 `ValueError`
    - 关键词 / Keywords:
      identity|linear|projection|deterministic|init|synthetic|benchmark|torch|weights|恒等初始化
    """
    if linear.in_features != linear.out_features:
        raise ValueError("Identity initialization requires a square linear layer.")
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight.copy_(torch.eye(linear.in_features, dtype=linear.weight.dtype))
        if linear.bias is not None:
            linear.bias.zero_()


def _route_one_hot(slot_id: int, slots: int, route_scale: float) -> torch.Tensor:
    vec = torch.zeros(slots, dtype=torch.float32)
    vec[slot_id] = route_scale
    return vec


def _symbol_one_hot(symbol_id: int, count: int, scale: float) -> torch.Tensor:
    vec = torch.zeros(count, dtype=torch.float32)
    vec[symbol_id] = scale
    return vec


def build_token_vector(
    key_id: int | None,
    value_id: int | None,
    *,
    slots: int,
    key_count: int,
    value_count: int,
    route_scale: float,
    key_scale: float,
    value_scale: float,
    filler_bias: float = -6.0,
    pattern_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create one synthetic token vector in the shared probing space.

    中文说明:
    - 调用方 / Called by: `build_exact_recall_cases`, `build_error_override_cases`,
      `build_anti_fixation_cases`
    - 调用对象 / Calls: `_route_one_hot`, `_symbol_one_hot`, `torch.full`, `torch.cat`
    - 作用 / Purpose: 统一生成 fact/query/filler token，复用 route-key-value 三段空间编码
    - 变量 / Variables:
      `key_id/value_id` 为离散键值编号；`route_scale/key_scale/value_scale`
      控制路由、精确键和值信号强度；`pattern_bias` 用于反执念实验加入共享模式偏置
    - 接入 / Integration: 任何新 probe 只要遵循同一张量空间即可复用现有模型适配器
    - 错误处理 / Error handling: `key_id/value_id` 为 `None` 时分别视为缺失键/值信号
    - 关键词 / Keywords:
      token_vector|route|key|value|filler|synthetic|encoding|streaming|probe|向量构造
    """
    route = torch.full((slots,), filler_bias, dtype=torch.float32)
    key_part = torch.full((key_count,), filler_bias, dtype=torch.float32)
    value_part = torch.full((value_count,), filler_bias, dtype=torch.float32)
    if key_id is not None:
        route[:] = 0.0
        route += _route_one_hot(key_id % slots, slots, route_scale)
        key_part[:] = 0.0
        key_part += _symbol_one_hot(key_id, key_count, key_scale)
    if value_id is not None:
        value_part[:] = 0.0
        value_part += _symbol_one_hot(value_id, value_count, value_scale)
    if pattern_bias is not None:
        key_part = key_part + pattern_bias.to(dtype=torch.float32)
    return torch.cat([route, key_part, value_part], dim=0)


def _generate_chunk(
    case: DiagnosticCase,
    start: int,
    end: int,
    filler_vector: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    chunk = filler_vector.repeat(end - start, 1)
    for position, token in case.special_tokens.items():
        if start <= position < end:
            chunk[position - start] = token
    return chunk.unsqueeze(0).to(device=device)


def _decode_prediction(
    output_token: torch.Tensor,
    candidate_value_ids: tuple[int, ...],
    slots: int,
    key_count: int,
    value_count: int,
) -> tuple[int, float]:
    value_slice = output_token[slots + key_count : slots + key_count + value_count]
    candidate_values = value_slice[list(candidate_value_ids)]
    best_idx = int(candidate_values.argmax().item())
    top_value = int(candidate_value_ids[best_idx])
    sorted_scores = torch.sort(candidate_values, descending=True).values
    margin = float((sorted_scores[0] - sorted_scores[1]).item()) if sorted_scores.numel() > 1 else 0.0
    return top_value, margin


def _build_dsra_layer(dim: int, slots: int) -> DSRA_Chunk_Layer:
    layer = DSRA_Chunk_Layer(
        dim=dim,
        K=slots,
        kr=1,
        use_orthogonal_update=True,
        use_bypass=False,
    )
    _set_identity_linear(layer.W_q)
    _set_identity_linear(layer.W_v)
    with torch.no_grad():
        layer.S_init.zero_()
        layer.W_n.weight.zero_()
        layer.W_n.bias.zero_()
        for slot_id in range(slots):
            layer.W_n.weight[slot_id, 1 + slot_id] = 8.0
    layer.eval()
    return layer


def _build_mhdsra2_layer(
    dim: int,
    slots: int,
    use_retrieval: bool,
    key_count: int | None = None,
    address_only_qk: bool = True,
    retrieval_tau: float = 8.0,
) -> MultiHeadDSRA2:
    """Build a deterministic MHDSRA2 layer for synthetic diagnostics.

    中文说明:
    - 调用方 / Called by: `run_case_for_model`, `tests.test_mhdsra2_isolation`
    - 调用对象 / Calls: `MHDSRA2Config`, `MultiHeadDSRA2`, `_set_identity_linear`, `torch.eye`
    - 作用 / Purpose: 构造确定性的单头 MHDSRA2；诊断模式下 Q/K 只读取 route+key 地址段，
      V 保留完整 token，从而验证 correction overwrite 时不被 value filler 干扰
    - 变量 / Variables:
      `dim` 为完整 probing 维度, `slots` 为路由槽位数, `key_count` 为 key 段长度,
      `address_only_qk` 控制 Q/K 是否只编码地址段, `use_retrieval` 控制 retrieval 分支门控,
      `retrieval_tau` 控制 paged recall softmax 锐度
    - 接入 / Integration: 新增 synthetic probe 时复用该 helper，并在 route/key/value 空间一致时传入 `key_count`
    - 错误处理 / Error handling: 依赖底层配置检查；`key_count=None` 时自动退回完整 identity Q/K
    - 关键词 / Keywords:
      mhdsra2|diagnostic|address_only_qk|route|key|value|identity|correction|overwrite|诊断构造
    """
    cfg = MHDSRA2Config(
        dim=dim,
        heads=1,
        slots=slots,
        read_topk=1,
        write_topk=1,
        local_window=0,
        use_local=False,
        use_retrieval=use_retrieval,
        retrieval_tau=retrieval_tau,
        detach_state=True,
    )
    layer = MultiHeadDSRA2(cfg)
    with torch.no_grad():
        layer.qkv.weight.zero_()
        eye = torch.eye(dim, dtype=layer.qkv.weight.dtype)
        if address_only_qk and key_count is not None:
            address_dim = min(dim, slots + key_count)
            address_eye = torch.eye(address_dim, dtype=layer.qkv.weight.dtype)
            layer.qkv.weight[:address_dim, :address_dim] = address_eye
            layer.qkv.weight[dim : dim + address_dim, :address_dim] = address_eye
        else:
            layer.qkv.weight[:dim, :] = eye
            layer.qkv.weight[dim : 2 * dim, :] = eye
        layer.qkv.weight[2 * dim :, :] = eye
        _set_identity_linear(layer.out_proj)
        layer.slot_k_init.zero_()
        layer.slot_v_init.zero_()
        for slot_id in range(slots):
            layer.slot_k_init[0, slot_id, slot_id] = 6.0
        layer.token_write_gate.weight.zero_()
        layer.token_write_gate.bias.fill_(6.0)
        layer.fuse_gate.weight.zero_()
        layer.fuse_gate.bias[:] = torch.tensor(
            [1.5, -6.0, 3.0 if use_retrieval else -6.0],
            dtype=layer.fuse_gate.bias.dtype,
        )
    layer.eval()
    return layer


def _build_sliding_window_layer(dim: int, window_size: int) -> SlidingWindowAttentionChunkLayer:
    layer = SlidingWindowAttentionChunkLayer(dim=dim, window_size=window_size)
    _set_identity_linear(layer.q_proj)
    _set_identity_linear(layer.k_proj)
    _set_identity_linear(layer.v_proj)
    _set_identity_linear(layer.out_proj)
    layer.eval()
    return layer


def _build_linear_attention_layer(dim: int) -> LinearAttentionChunkLayer:
    layer = LinearAttentionChunkLayer(dim=dim)
    _set_identity_linear(layer.q_proj)
    _set_identity_linear(layer.k_proj)
    _set_identity_linear(layer.v_proj)
    _set_identity_linear(layer.out_proj)
    layer.eval()
    return layer


def run_case_for_model(
    case: DiagnosticCase,
    model_name: str,
    *,
    slots: int,
    key_count: int,
    value_count: int,
    chunk_size: int,
    device: torch.device,
    page_size: int,
    retrieved_top_pages: int,
    retrieved_max_tokens: int,
    retrieval_tau: float = 8.0,
) -> dict:
    """Execute one diagnostic case for a specific memory family.

    中文说明:
    - 调用方 / Called by: `run_diagnostic_suite`
    - 调用对象 / Calls:
      `_build_dsra_layer`, `_build_mhdsra2_layer`, `_build_sliding_window_layer`,
      `_build_linear_attention_layer`, `_generate_chunk`, `_decode_prediction`,
      `PagedExactMemory.retrieve`
    - 作用 / Purpose: 对单个 A/B/C 样例执行流式前向，输出预测值和可诊断内部统计
    - 变量 / Variables:
      `case` 为样例定义, `model_name` 为模型族名称, `chunk_size/page_size`
      控制流式分块与 paged recall, `retrieved_top_pages/max_tokens`
      控制外部精确召回规模, `retrieval_tau` 控制 MHDSRA2 retrieval softmax 温度
    - 接入 / Integration: 新模型族只需在本函数新增一个分支并返回统一字典
    - 错误处理 / Error handling: 未知 `model_name` 时抛出 `ValueError`
    - 关键词 / Keywords:
      run_case|streaming|paged_recall|prediction|diagnostic|model_family|aux_stats|benchmark|chunk|执行
    """
    filler_vector = build_token_vector(
        None,
        None,
        slots=slots,
        key_count=key_count,
        value_count=value_count,
        route_scale=0.0,
        key_scale=0.0,
        value_scale=0.0,
    )
    predicted_value = None
    confidence_margin = 0.0
    aux_summary = {}

    with torch.no_grad():
        if model_name == "dsra":
            layer = _build_dsra_layer(case.dim, slots).to(device)
            state = None
            bypass_kv = None
            for start in range(0, case.query_position, chunk_size):
                end = min(case.query_position, start + chunk_size)
                chunk = _generate_chunk(case, start, end, filler_vector, device)
                _, state, bypass_kv, _ = layer(chunk, S_prev=state, bypass_kv=bypass_kv)
            query_chunk = _generate_chunk(case, case.query_position, case.query_position + 1, filler_vector, device)
            output, _, _, _ = layer(query_chunk, S_prev=state, bypass_kv=bypass_kv)
            predicted_value, confidence_margin = _decode_prediction(
                output[0, -1].cpu(),
                case.candidate_value_ids,
                slots,
                key_count,
                value_count,
            )
        elif model_name in {"mhdsra2_without_paged_recall", "mhdsra2_with_paged_recall"}:
            use_retrieval = model_name == "mhdsra2_with_paged_recall"
            layer = _build_mhdsra2_layer(
                case.dim,
                slots,
                use_retrieval=use_retrieval,
                key_count=key_count,
                retrieval_tau=retrieval_tau,
            ).to(device)
            state = None
            memory = PagedExactMemory(page_size=page_size, dtype=torch.float32) if use_retrieval else None
            for start in range(0, case.query_position, chunk_size):
                end = min(case.query_position, start + chunk_size)
                chunk = _generate_chunk(case, start, end, filler_vector, device)
                output, state, aux = layer(chunk, state=state, return_aux=True)
                if use_retrieval and memory is not None:
                    qkv = layer.qkv(chunk)
                    _, k, v = qkv.chunk(3, dim=-1)
                    memory.append(layer._to_heads(k), layer._to_heads(v))
                aux_summary = {
                    "slot_usage_mean": float(aux["slot_usage"].mean().item()),
                    "slot_confidence_mean": float(aux["slot_confidence"].mean().item()),
                }
                if aux.get("write_stats") is not None:
                    aux_summary.update(
                        {
                            "forget_gate_mean": float(aux["write_stats"]["forget_gate_mean"].item()),
                            "conflict_mean": float(aux["write_stats"]["conflict_mean"].item()),
                            "write_gate_mean": float(aux["write_stats"]["write_gate_mean"].item()),
                        }
                    )
            query_chunk = _generate_chunk(case, case.query_position, case.query_position + 1, filler_vector, device)
            retrieved_k = None
            retrieved_v = None
            if use_retrieval and memory is not None:
                q_only = layer._to_heads(layer.qkv(query_chunk).chunk(3, dim=-1)[0])
                retrieved_k, retrieved_v, retrieved_positions = memory.retrieve(
                    q_only,
                    top_pages=retrieved_top_pages,
                    max_tokens=retrieved_max_tokens,
                    device=device,
                )
                aux_summary["retrieved_token_count"] = int(0 if retrieved_positions is None else retrieved_positions.numel())
            output, _, aux = layer(
                query_chunk,
                state=state,
                retrieved_k=retrieved_k,
                retrieved_v=retrieved_v,
                return_aux=True,
            )
            gates_tensor = aux["gates_mean"].detach().float()
            if gates_tensor.dim() > 1:
                gates_tensor = gates_tensor.mean(dim=0)
            gates = gates_tensor.cpu().tolist()
            aux_summary["slot_gate_mean"] = float(gates[0])
            aux_summary["retrieval_gate_mean"] = float(gates[2] if len(gates) > 2 else 0.0)
            if aux.get("write_stats") is not None:
                aux_summary.update(
                    {
                        "forget_gate_mean": float(aux["write_stats"]["forget_gate_mean"].item()),
                        "conflict_mean": float(aux["write_stats"]["conflict_mean"].item()),
                        "write_gate_mean": float(aux["write_stats"]["write_gate_mean"].item()),
                    }
                )
            predicted_value, confidence_margin = _decode_prediction(
                output[0, -1].cpu(),
                case.candidate_value_ids,
                slots,
                key_count,
                value_count,
            )
        elif model_name == "sliding_window_attention":
            layer = _build_sliding_window_layer(case.dim, window_size=max(chunk_size * 2, 64)).to(device)
            kv_cache = None
            for start in range(0, case.query_position, chunk_size):
                end = min(case.query_position, start + chunk_size)
                chunk = _generate_chunk(case, start, end, filler_vector, device)
                _, _, kv_cache, _ = layer(chunk, bypass_kv=kv_cache)
            query_chunk = _generate_chunk(case, case.query_position, case.query_position + 1, filler_vector, device)
            output, _, _, _ = layer(query_chunk, bypass_kv=kv_cache)
            predicted_value, confidence_margin = _decode_prediction(
                output[0, -1].cpu(),
                case.candidate_value_ids,
                slots,
                key_count,
                value_count,
            )
        elif model_name == "linear_attention":
            layer = _build_linear_attention_layer(case.dim).to(device)
            state = None
            for start in range(0, case.query_position, chunk_size):
                end = min(case.query_position, start + chunk_size)
                chunk = _generate_chunk(case, start, end, filler_vector, device)
                _, state, _, _ = layer(chunk, S_prev=state)
            query_chunk = _generate_chunk(case, case.query_position, case.query_position + 1, filler_vector, device)
            output, _, _, _ = layer(query_chunk, S_prev=state)
            predicted_value, confidence_margin = _decode_prediction(
                output[0, -1].cpu(),
                case.candidate_value_ids,
                slots,
                key_count,
                value_count,
            )
        else:
            raise ValueError(f"Unsupported diagnostic model: {model_name}")

    return {
        "predicted_value_id": predicted_value,
        "target_value_id": case.target_value_id,
        "is_correct": float(predicted_value == case.target_value_id),
        "confidence_margin": float(confidence_margin),
        "aux": aux_summary,
    }


def build_exact_recall_cases(args: argparse.Namespace, dim: int) -> list[DiagnosticCase]:
    slots = args.diagnostic_slots
    key_count = args.diagnostic_key_count
    value_count = args.diagnostic_value_count
    seq_len = args.diagnostic_exact_seq_len
    query_position = seq_len - 1
    positions = {
        "early": max(8, seq_len // 20),
        "middle": max(8, seq_len // 2),
        "late": max(8, seq_len - max(args.diagnostic_exact_fact_spacing * 2, 32)),
    }
    cases = []
    for position_name, target_position in positions.items():
        special_tokens = {}
        for fact_position in range(0, query_position, args.diagnostic_exact_fact_spacing):
            key_id = (fact_position // args.diagnostic_exact_fact_spacing) % key_count
            value_id = key_id % value_count
            special_tokens[fact_position] = build_token_vector(
                key_id,
                value_id,
                slots=slots,
                key_count=key_count,
                value_count=value_count,
                route_scale=6.0,
                key_scale=2.0,
                value_scale=3.0,
            )
        target_key = slots
        target_value = slots % value_count
        special_tokens[target_position] = build_token_vector(
            target_key,
            target_value,
            slots=slots,
            key_count=key_count,
            value_count=value_count,
            route_scale=6.0,
            key_scale=2.0,
            value_scale=3.0,
        )
        special_tokens[query_position] = build_token_vector(
            target_key,
            None,
            slots=slots,
            key_count=key_count,
            value_count=value_count,
            route_scale=6.0,
            key_scale=2.0,
            value_scale=0.0,
        )
        cases.append(
            DiagnosticCase(
                suite="diagnostic_a_exact_recall",
                case_id=f"exact_recall.{position_name}",
                seq_len=seq_len,
                query_position=query_position,
                dim=dim,
                target_value_id=target_value,
                candidate_value_ids=tuple(range(value_count)),
                special_tokens=special_tokens,
                metadata={
                    "target_position": target_position,
                    "position_bucket": position_name,
                    "fact_spacing": args.diagnostic_exact_fact_spacing,
                },
            )
        )
    return cases


def build_error_override_cases(args: argparse.Namespace, dim: int) -> list[DiagnosticCase]:
    slots = args.diagnostic_slots
    key_count = args.diagnostic_key_count
    value_count = args.diagnostic_value_count
    seq_len = args.diagnostic_override_seq_len
    query_position = seq_len - 1
    key_id = slots // 2
    old_value = 0
    new_value = 1
    cases = []
    for gap in args.diagnostic_override_gap_grid:
        correction_position = max(16, query_position - gap)
        old_position = max(4, correction_position // 4)
        special_tokens = {
            old_position: build_token_vector(
                key_id,
                old_value,
                slots=slots,
                key_count=key_count,
                value_count=value_count,
                route_scale=6.0,
                key_scale=2.0,
                value_scale=3.0,
            ),
            correction_position: build_token_vector(
                key_id,
                new_value,
                slots=slots,
                key_count=key_count,
                value_count=value_count,
                route_scale=6.0,
                key_scale=2.0,
                value_scale=3.0,
            ),
            query_position: build_token_vector(
                key_id,
                None,
                slots=slots,
                key_count=key_count,
                value_count=value_count,
                route_scale=6.0,
                key_scale=2.0,
                value_scale=0.0,
            ),
        }
        cases.append(
            DiagnosticCase(
                suite="diagnostic_b_error_override",
                case_id=f"error_override.gap_{gap}",
                seq_len=seq_len,
                query_position=query_position,
                dim=dim,
                target_value_id=new_value,
                candidate_value_ids=(old_value, new_value),
                special_tokens=special_tokens,
                metadata={"gap": gap, "old_value_id": old_value, "new_value_id": new_value},
            )
        )
    return cases


def build_anti_fixation_cases(args: argparse.Namespace, dim: int) -> list[DiagnosticCase]:
    slots = args.diagnostic_slots
    key_count = args.diagnostic_key_count
    value_count = args.diagnostic_value_count
    seq_len = args.diagnostic_fixation_seq_len
    query_position = seq_len - 1
    target_key = slots + 3
    shared_bias = torch.zeros(key_count, dtype=torch.float32)
    shared_bias[target_key] = 0.75
    cases = []
    for distractor_count in args.diagnostic_fixation_distractor_grid:
        special_tokens = {}
        step = max(2, query_position // max(distractor_count + 2, 2))
        wrong_value = 0
        right_value = 1
        for distractor_idx in range(distractor_count):
            key_id = (target_key + distractor_idx + 1) % key_count
            special_tokens[min(distractor_idx * step, query_position - 2)] = build_token_vector(
                key_id,
                wrong_value,
                slots=slots,
                key_count=key_count,
                value_count=value_count,
                route_scale=6.0,
                key_scale=0.6,
                value_scale=3.0,
                pattern_bias=shared_bias,
            )
        exception_position = max(8, query_position - step * 2)
        special_tokens[exception_position] = build_token_vector(
            target_key,
            right_value,
            slots=slots,
            key_count=key_count,
            value_count=value_count,
            route_scale=6.0,
            key_scale=1.8,
            value_scale=3.0,
            pattern_bias=shared_bias,
        )
        special_tokens[query_position] = build_token_vector(
            target_key,
            None,
            slots=slots,
            key_count=key_count,
            value_count=value_count,
            route_scale=6.0,
            key_scale=1.8,
            value_scale=0.0,
            pattern_bias=shared_bias,
        )
        cases.append(
            DiagnosticCase(
                suite="diagnostic_c_anti_fixation",
                case_id=f"anti_fixation.distractors_{distractor_count}",
                seq_len=seq_len,
                query_position=query_position,
                dim=dim,
                target_value_id=right_value,
                candidate_value_ids=(wrong_value, right_value),
                special_tokens=special_tokens,
                metadata={"distractor_count": distractor_count, "majority_value_id": wrong_value},
            )
        )
    return cases


def _aggregate_suite_metrics(suite_name: str, case_results: list[dict]) -> dict:
    per_model = {model_name: {} for model_name in MODEL_ORDER}
    for model_name in MODEL_ORDER:
        valid_model_results = [
            case["models"][model_name]
            for case in case_results
            if case["models"][model_name]["is_correct"] is not None
        ]
        exact_scores = [result["is_correct"] for result in valid_model_results]
        confidence_values = [
            result["confidence_margin"]
            for result in valid_model_results
            if result["confidence_margin"] is not None
        ]
        oom_count = sum(1 for case in case_results if case["models"][model_name].get("error") == "oom")
        per_model[model_name]["successful_cases"] = float(len(valid_model_results))
        per_model[model_name]["oom_cases"] = float(oom_count)
        per_model[model_name]["confidence_margin"] = (
            float(sum(confidence_values) / len(confidence_values)) if confidence_values else None
        )
        if suite_name == "diagnostic_a_exact_recall":
            per_model[model_name]["exact_match_rate"] = (
                float(sum(exact_scores) / len(exact_scores)) if exact_scores else None
            )
        elif suite_name == "diagnostic_b_error_override":
            stale_hits = [
                1.0
                if case["models"][model_name]["predicted_value_id"] == case["metadata"]["old_value_id"]
                else 0.0
                for case in case_results
                if case["models"][model_name]["predicted_value_id"] is not None
            ]
            per_model[model_name]["latest_fact_accuracy"] = (
                float(sum(exact_scores) / len(exact_scores)) if exact_scores else None
            )
            per_model[model_name]["stale_fact_rate"] = (
                float(sum(stale_hits) / len(stale_hits)) if stale_hits else None
            )
            aux_values = [
                case["models"][model_name]["aux"].get("forget_gate_mean")
                for case in case_results
                if "forget_gate_mean" in case["models"][model_name]["aux"]
            ]
            per_model[model_name]["forget_gate_mean"] = float(sum(aux_values) / max(len(aux_values), 1)) if aux_values else None
        elif suite_name == "diagnostic_c_anti_fixation":
            trap_hits = [
                1.0
                if case["models"][model_name]["predicted_value_id"] == case["metadata"]["majority_value_id"]
                else 0.0
                for case in case_results
                if case["models"][model_name]["predicted_value_id"] is not None
            ]
            per_model[model_name]["counterexample_accuracy"] = (
                float(sum(exact_scores) / len(exact_scores)) if exact_scores else None
            )
            per_model[model_name]["majority_trap_rate"] = (
                float(sum(trap_hits) / len(trap_hits)) if trap_hits else None
            )
    return per_model


def _build_model_table(title: str, metrics_by_model: dict) -> dict:
    metric_names = []
    for model_metrics in metrics_by_model.values():
        for metric_name in model_metrics:
            if metric_name not in metric_names:
                metric_names.append(metric_name)
    columns = ["Model", *metric_names]
    rows = []
    for model_name in MODEL_ORDER:
        row = [MODEL_LABELS[model_name]]
        for metric_name in metric_names:
            value = metrics_by_model[model_name].get(metric_name)
            row.append("-" if value is None else f"{value:.4f}")
        rows.append(row)
    return {"title": title, "columns": columns, "rows": rows}


def _build_case_table(case_results: list[dict]) -> dict:
    columns = ["Case", *[MODEL_LABELS[name] for name in MODEL_ORDER]]
    rows = []
    for case in case_results:
        row = [case["case_id"]]
        for model_name in MODEL_ORDER:
            score = case["models"][model_name]["is_correct"]
            error_kind = case["models"][model_name].get("error")
            if error_kind == "oom":
                row.append("OOM")
            elif score is None:
                row.append("-")
            else:
                row.append(f"{score:.0f}")
        rows.append(row)
    return {"title": "Per-case exact success", "columns": columns, "rows": rows}


def _is_oom_error(exc: BaseException) -> bool:
    """Check whether one execution error is an out-of-memory failure.

    中文说明:
    - 调用方 / Called by: `_run_case_for_model_with_oom_guard`
    - 调用对象 / Calls: `str`
    - 作用 / Purpose: 统一识别 CUDA/加速器/通用 RuntimeError 中的 OOM 场景
    - 变量 / Variables: `exc` 为待识别异常对象
    - 接入 / Integration: 其他需要 benchmark 容错的脚本也可直接复用本函数
    - 错误处理 / Error handling: 无法识别时返回 `False`，交由上层继续抛出原异常
    - 关键词 / Keywords:
      oom|out_of_memory|runtimeerror|cuda|accelerator|guard|benchmark|exception|detect|内存溢出
    """
    message = str(exc).lower()
    return "out of memory" in message or ("cuda error" in message and "memory" in message)


def _cleanup_after_oom() -> None:
    """Release Python and CUDA caches after an OOM branch.

    中文说明:
    - 调用方 / Called by: `_run_case_for_model_with_oom_guard`
    - 调用对象 / Calls: `gc.collect`, `torch.cuda.empty_cache`
    - 作用 / Purpose: 在诊断中出现 OOM 后尽快回收缓存，降低后续模型继续失败概率
    - 变量 / Variables: 无
    - 接入 / Integration: 其他长上下文 benchmark 若需要失败后继续执行，也可复用
    - 错误处理 / Error handling: 无 CUDA 环境时自动跳过显存清理，不抛出异常
    - 关键词 / Keywords:
      cleanup|oom|gc|cuda_cache|empty_cache|benchmark|memory|recovery|guard|清理
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_oom_result(case: DiagnosticCase, model_name: str, exc: BaseException) -> dict:
    """Create one normalized missing-result payload for an OOM model run.

    中文说明:
    - 调用方 / Called by: `_run_case_for_model_with_oom_guard`
    - 调用对象 / Calls: `str`
    - 作用 / Purpose: 把 OOM 转为统一缺失记录，保证后续聚合与报告链路不中断
    - 变量 / Variables:
      `case` 当前样例, `model_name` 当前模型名, `exc` 为原始 OOM 异常
    - 接入 / Integration: 统一结果结构可直接进入 `case_results` 和 `model_tables`
    - 错误处理 / Error handling: 仅保存简短错误文本，不重新抛出异常
    - 关键词 / Keywords:
      oom_result|missing|normalized|diagnostic|case|model|report|guard|payload|占位结果
    """
    return {
        "predicted_value_id": None,
        "target_value_id": case.target_value_id,
        "is_correct": None,
        "confidence_margin": None,
        "aux": {},
        "error": "oom",
        "error_message": f"{model_name}: {type(exc).__name__}: {exc}",
    }


def _run_case_for_model_with_oom_guard(
    case: DiagnosticCase,
    model_name: str,
    *,
    slots: int,
    key_count: int,
    value_count: int,
    chunk_size: int,
    device: torch.device,
    page_size: int,
    retrieved_top_pages: int,
    retrieved_max_tokens: int,
    retrieval_tau: float = 8.0,
) -> dict:
    """Execute one case with OOM capture so the benchmark can continue.

    中文说明:
    - 调用方 / Called by: `run_diagnostic_suite`
    - 调用对象 / Calls:
      `run_case_for_model`, `_is_oom_error`, `_cleanup_after_oom`, `_build_oom_result`
    - 作用 / Purpose: 包装逐模型执行，单模型 OOM 时只写缺失结果而不中断整套实验
    - 变量 / Variables: 与 `run_case_for_model` 相同，外加 OOM 容错分支
    - 接入 / Integration: 作为五模型循环的唯一入口，避免各处重复 try/except
    - 错误处理 / Error handling: OOM 转缺失结果；非 OOM 异常保持原样抛出
    - 关键词 / Keywords:
      oom_guard|safe_run|diagnostic|continue|benchmark|model_loop|exception|recovery|runner|容错执行
    """
    accelerator_error = getattr(torch, "AcceleratorError", None)
    try:
        return run_case_for_model(
            case,
            model_name,
            slots=slots,
            key_count=key_count,
            value_count=value_count,
            chunk_size=chunk_size,
            device=device,
            page_size=page_size,
            retrieved_top_pages=retrieved_top_pages,
            retrieved_max_tokens=retrieved_max_tokens,
            retrieval_tau=retrieval_tau,
        )
    except torch.cuda.OutOfMemoryError as exc:
        _cleanup_after_oom()
        return _build_oom_result(case, model_name, exc)
    except RuntimeError as exc:
        if not _is_oom_error(exc):
            raise
        _cleanup_after_oom()
        return _build_oom_result(case, model_name, exc)
    except Exception as exc:
        if accelerator_error is None or not isinstance(exc, accelerator_error) or not _is_oom_error(exc):
            raise
        _cleanup_after_oom()
        return _build_oom_result(case, model_name, exc)


def run_diagnostic_suite(args: argparse.Namespace, suite_name: str, cases: list[DiagnosticCase]) -> dict:
    """Run one diagnostic suite and return one report-ready section.

    中文说明:
    - 调用方 / Called by: `run_diagnostic_benchmarks`
    - 调用对象 / Calls: `run_case_for_model`, `_aggregate_suite_metrics`,
      `_build_model_table`, `_build_case_table`, `build_benchmark_comparison_row`
    - 作用 / Purpose: 执行 A/B/C 中任一套件，整理 pairwise rows 和五模型汇总表
    - 变量 / Variables:
      `suite_name` 为诊断套件名, `cases` 为套件内样例列表,
      `case_results` 保存逐 case 多模型输出
    - 接入 / Integration: 返回结构直接交给现有 `build_benchmark_payload` / 报告链路
    - 错误处理 / Error handling: 单模型 OOM 会转成缺失结果继续执行；非 OOM 异常直接上抛
    - 关键词 / Keywords:
      diagnostic_suite|section|rows|model_table|pairwise|streaming|benchmark|report|cases|套件执行
    """
    device = torch.device(args.diagnostic_device)
    retrieval_tau = float(getattr(args, "diagnostic_retrieval_tau", 8.0))
    case_results = []
    for case in cases:
        model_results = {}
        for model_name in MODEL_ORDER:
            model_results[model_name] = _run_case_for_model_with_oom_guard(
                case,
                model_name,
                slots=args.diagnostic_slots,
                key_count=args.diagnostic_key_count,
                value_count=args.diagnostic_value_count,
                chunk_size=args.diagnostic_chunk_size,
                device=device,
                page_size=args.diagnostic_page_size,
                retrieved_top_pages=args.diagnostic_retrieved_top_pages,
                retrieved_max_tokens=args.diagnostic_retrieved_max_tokens,
                retrieval_tau=retrieval_tau,
            )
        case_results.append({"case_id": case.case_id, "metadata": case.metadata, "models": model_results})

    metrics_by_model = _aggregate_suite_metrics(suite_name, case_results)
    primary_metric = PRIMARY_METRIC_BY_SUITE[suite_name]
    higher_is_better = primary_metric not in LOWER_IS_BETTER_METRICS
    rows = []
    oom_counts = {
        model_name: sum(1 for case in case_results if case["models"][model_name].get("error") == "oom")
        for model_name in MODEL_ORDER
    }
    for mh_variant in ("mhdsra2_without_paged_recall", "mhdsra2_with_paged_recall"):
        rows.append(
            build_benchmark_comparison_row(
                suite=suite_name,
                task="aggregate",
                split="overall",
                metric=f"{primary_metric}.{mh_variant}",
                dsra_value=metrics_by_model["dsra"].get(primary_metric),
                mhdsra2_value=metrics_by_model[mh_variant].get(primary_metric),
                higher_is_better=higher_is_better,
                notes=(
                    f"oom(dsra={oom_counts['dsra']}, {mh_variant}={oom_counts[mh_variant]})"
                    if oom_counts["dsra"] or oom_counts[mh_variant]
                    else ""
                ),
                metadata={"compared_variant": mh_variant, "cases": [case.case_id for case in cases]},
            )
        )

    title_map = {
        "diagnostic_a_exact_recall": "Diagnostic A - Exact Recall",
        "diagnostic_b_error_override": "Diagnostic B - Error Override",
        "diagnostic_c_anti_fixation": "Diagnostic C - Anti-fixation",
    }
    description_map = {
        "diagnostic_a_exact_recall": "Streams sparse fact tokens through long filler contexts and checks whether the queried value is recovered exactly.",
        "diagnostic_b_error_override": "Writes an old fact, then a correction for the same key, and measures whether the latest fact overrides stale memory.",
        "diagnostic_c_anti_fixation": "Builds a dominant wrong pattern plus one rare counterexample, then checks whether the model resists majority fixation.",
    }
    return {
        "title": title_map[suite_name],
        "description": description_map[suite_name],
        "rows": rows,
        "model_tables": [
            _build_model_table("Mean metrics by model", metrics_by_model),
            _build_case_table(case_results),
        ],
        "diagnostic_cases": case_results,
    }


def run_diagnostic_benchmarks(args: argparse.Namespace) -> list[dict]:
    """Run all A/B/C diagnostics and return benchmark sections.

    中文说明:
    - 调用方 / Called by: `main`, `scripts.next_round_benchmark_runner.run_next_round_benchmark`
    - 调用对象 / Calls:
      `build_exact_recall_cases`, `build_error_override_cases`,
      `build_anti_fixation_cases`, `run_diagnostic_suite`
    - 作用 / Purpose: 统一组织三组诊断实验，供独立脚本或 next-round runner 直接复用
    - 变量 / Variables:
      `dim` 为共享 probe 维度；三类 `cases` 分别表示 A/B/C 诊断套件
    - 接入 / Integration: 返回值是现有报告链路可消费的 `section` 列表
    - 错误处理 / Error handling: 套件内部单模型 OOM 会被容错；非 OOM 异常保持真实失败信号
    - 关键词 / Keywords:
      diagnostics|a_b_c|sections|runner|report_chain|streaming|probe|mhdsra2|dsra|统一入口
    """
    dim = args.diagnostic_slots + args.diagnostic_key_count + args.diagnostic_value_count
    sections = []
    sections.append(run_diagnostic_suite(args, "diagnostic_a_exact_recall", build_exact_recall_cases(args, dim)))
    sections.append(run_diagnostic_suite(args, "diagnostic_b_error_override", build_error_override_cases(args, dim)))
    sections.append(run_diagnostic_suite(args, "diagnostic_c_anti_fixation", build_anti_fixation_cases(args, dim)))
    return sections


def add_diagnostic_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach A/B/C diagnostic CLI arguments to an existing parser.

    中文说明:
    - 调用方 / Called by: `build_parser`,
      `scripts.next_round_benchmark_runner.build_parser`
    - 调用对象 / Calls: `ArgumentParser.add_argument`
    - 作用 / Purpose: 把诊断实验参数注册到独立脚本与 next-round runner，共享同一套默认值
    - 变量 / Variables: `parser` 为待扩展的命令行解析器
    - 接入 / Integration: 其他 runner 若要复用同一诊断脚本，只需先调用本函数
    - 错误处理 / Error handling: 参数校验继续交由 argparse 处理
    - 关键词 / Keywords:
      parser|cli|diagnostic|arguments|shared_defaults|runner|benchmark|config|args|命令行扩展
    """
    parser.add_argument("--diagnostic-device", type=str, default="cpu")
    parser.add_argument("--diagnostic-slots", type=int, default=16)
    parser.add_argument("--diagnostic-key-count", type=int, default=64)
    parser.add_argument("--diagnostic-value-count", type=int, default=64)
    parser.add_argument("--diagnostic-chunk-size", type=int, default=256)
    parser.add_argument("--diagnostic-page-size", type=int, default=128)
    parser.add_argument("--diagnostic-retrieved-top-pages", type=int, default=4)
    parser.add_argument("--diagnostic-retrieved-max-tokens", type=int, default=64)
    parser.add_argument("--diagnostic-retrieval-tau", type=float, default=8.0)
    parser.add_argument("--diagnostic-exact-seq-len", type=int, default=2_000_000)
    parser.add_argument("--diagnostic-exact-fact-spacing", type=int, default=1024)
    parser.add_argument("--diagnostic-override-seq-len", type=int, default=16384)
    parser.add_argument("--diagnostic-override-gap-grid", nargs="+", type=int, default=[128, 1024, 4096])
    parser.add_argument("--diagnostic-fixation-seq-len", type=int, default=65536)
    parser.add_argument("--diagnostic-fixation-distractor-grid", nargs="+", type=int, default=[16, 64, 256])
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_diagnostic_cli_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    return {"sections": run_diagnostic_benchmarks(args)}


if __name__ == "__main__":
    main()
