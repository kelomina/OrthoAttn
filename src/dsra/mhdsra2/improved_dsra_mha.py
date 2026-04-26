"""Multi-head DSRA-v2 prototype.

This file implements a streaming, chunk-wise attention module designed for very
long contexts. It keeps only a bounded local KV cache and a fixed number of
per-head global slots on GPU. Optional exact retrieval KV can be supplied by an
external CPU/NVMe memory index.

The implementation is intentionally conservative: it removes the O(K^3)
orthogonal inverse in the original DSRA and replaces dense top-k distributions
with gather/scatter top-k operations so the main buffers scale with
O(B * H * C * K) per chunk, not O(B * H * T^2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MHDSRA2Config:
    dim: int
    heads: int = 8
    slots: int = 128
    read_topk: int = 8
    write_topk: int = 4
    local_window: int = 512
    use_local: bool = True
    use_retrieval: bool = True
    tau_init: float = 8.0
    tau_write_init: float = 4.0
    eta: float = 0.25
    max_update: float = 0.50
    forget_base: float = 0.001
    forget_conflict: float = 0.20
    forget_age: float = 0.0002
    usage_decay: float = 0.995
    conf_decay: float = 0.999
    usage_prior: float = 0.25
    age_write_bias: float = 0.02
    conf_read_bias: float = 0.50
    age_read_penalty: float = 0.005
    detach_state: bool = True
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.dim % self.heads != 0:
            raise ValueError("dim must be divisible by heads")
        if self.read_topk < 1 or self.write_topk < 1:
            raise ValueError("top-k values must be positive")
        if self.slots < 1:
            raise ValueError("slots must be positive")


@dataclass
class MHDSRA2State:
    slot_k: torch.Tensor
    slot_v: torch.Tensor
    age: torch.Tensor
    usage: torch.Tensor
    confidence: torch.Tensor
    local_k: Optional[torch.Tensor] = None
    local_v: Optional[torch.Tensor] = None
    position: int = 0


class MultiHeadDSRA2(nn.Module):
    """Streaming multi-head DSRA-v2 layer.

    Forward input shape:  x [B, C, D]
    Forward output shape: y [B, C, D]

    State shapes are bounded by slots K and local_window W. For 2M-token
    sequences, call this module repeatedly on chunks and carry the returned
    state forward.
    """

    def __init__(self, cfg: MHDSRA2Config):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.heads = cfg.heads
        self.d_head = cfg.dim // cfg.heads
        self.slots = cfg.slots

        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.slot_k_init = nn.Parameter(
            torch.randn(cfg.heads, cfg.slots, self.d_head) / (self.d_head**0.5)
        )
        self.slot_v_init = nn.Parameter(
            torch.randn(cfg.heads, cfg.slots, self.d_head) / (self.d_head**0.5)
        )

        self.token_write_gate = nn.Linear(self.d_head, 1)
        self.fuse_gate = nn.Linear(self.d_head, 3)

        self.log_tau_read = nn.Parameter(torch.log(torch.tensor(float(cfg.tau_init))))
        self.log_tau_write = nn.Parameter(torch.log(torch.tensor(float(cfg.tau_write_init))))

    def init_state(self, batch_size: int, device=None, dtype=None) -> MHDSRA2State:
        cfg = self.cfg
        k = F.normalize(self.slot_k_init, dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        v = self.slot_v_init.unsqueeze(0).expand(batch_size, -1, -1, -1)
        k = k.to(device=device, dtype=dtype)
        v = v.to(device=device, dtype=dtype)
        zeros = torch.zeros(batch_size, cfg.heads, cfg.slots, device=device, dtype=torch.float32)
        conf = torch.full_like(zeros, 0.5)
        return MHDSRA2State(k.contiguous(), v.contiguous(), zeros, zeros.clone(), conf)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.heads, self.d_head).transpose(1, 2).contiguous()

    def _from_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    @staticmethod
    def _gather_slots(slots: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        b, h, k, d = slots.shape
        t, r = idx.shape[2], idx.shape[3]
        expanded = slots.unsqueeze(2).expand(b, h, t, k, d)
        gather_idx = idx.unsqueeze(-1).expand(b, h, t, r, d)
        return torch.gather(expanded, dim=3, index=gather_idx)

    @staticmethod
    def _scatter_mass(idx: torch.Tensor, weights: torch.Tensor, slots: int) -> torch.Tensor:
        b, h, t, r = idx.shape
        idx_flat = idx.reshape(b * h, t * r, 1)
        src = weights.reshape(b * h, t * r, 1).to(dtype=torch.float32)
        out = torch.zeros(b * h, slots, 1, device=weights.device, dtype=torch.float32)
        out.scatter_add_(1, idx_flat, src)
        return out.view(b, h, slots)

    @staticmethod
    def _scatter_values(
        idx: torch.Tensor, weights: torch.Tensor, values: torch.Tensor, slots: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, t, r = idx.shape
        d = values.shape[-1]
        idx_flat = idx.reshape(b * h, t * r)
        src = (weights.unsqueeze(-1) * values.unsqueeze(3)).reshape(b * h, t * r, d)
        out = torch.zeros(b * h, slots, d, device=values.device, dtype=values.dtype)
        out.scatter_add_(1, idx_flat.unsqueeze(-1).expand(-1, -1, d), src)

        mass_src = weights.reshape(b * h, t * r, 1).to(dtype=torch.float32)
        mass = torch.zeros(b * h, slots, 1, device=values.device, dtype=torch.float32)
        mass.scatter_add_(1, idx_flat.unsqueeze(-1), mass_src)
        return out.view(b, h, slots, d), mass.view(b, h, slots, 1)

    @staticmethod
    def _causal_prefix_mask(
        t_q: int, t_k: int, prefix_len: int, device, dtype
    ) -> torch.Tensor:
        q_pos = torch.arange(t_q, device=device).unsqueeze(1) + prefix_len
        k_pos = torch.arange(t_k, device=device).unsqueeze(0)
        mask = torch.zeros(t_q, t_k, device=device, dtype=dtype)
        mask = mask.masked_fill(k_pos > q_pos, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)

    def _slot_read(
        self, q: torch.Tensor, state: MHDSRA2State
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        slot_k = state.slot_k.to(dtype=q.dtype)
        slot_v = state.slot_v.to(dtype=q.dtype)
        qn = F.normalize(q, dim=-1)
        sk = F.normalize(slot_k, dim=-1)

        tau = self.log_tau_read.exp().to(dtype=q.dtype)
        logits = torch.einsum("bhtd,bhkd->bhtk", qn, sk) * tau
        logits = logits + cfg.conf_read_bias * state.confidence.to(dtype=q.dtype).unsqueeze(2)
        logits = logits - cfg.age_read_penalty * torch.log1p(state.age).to(dtype=q.dtype).unsqueeze(2)

        r = min(cfg.read_topk, cfg.slots)
        top_logits, top_idx = torch.topk(logits, r, dim=-1)
        probs = F.softmax(top_logits, dim=-1)
        selected_v = self._gather_slots(slot_v, top_idx)
        out = (probs.unsqueeze(-1) * selected_v).sum(dim=3)
        read_mass = self._scatter_mass(top_idx, probs, cfg.slots)
        return out, {
            "read_idx": top_idx,
            "read_probs": probs,
            "read_mass": read_mass,
            "read_logits_top": top_logits,
        }

    def _local_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state: MHDSRA2State
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        cfg = self.cfg
        if not cfg.use_local or cfg.local_window <= 0:
            return torch.zeros_like(q), None, None

        if state.local_k is not None and state.local_v is not None:
            prev_k = state.local_k.to(device=q.device, dtype=q.dtype)
            prev_v = state.local_v.to(device=q.device, dtype=q.dtype)
            if prev_k.shape[2] > cfg.local_window:
                prev_k = prev_k[:, :, -cfg.local_window :, :]
                prev_v = prev_v[:, :, -cfg.local_window :, :]
            k_cat = torch.cat([prev_k, k], dim=2)
            v_cat = torch.cat([prev_v, v], dim=2)
            prefix = prev_k.shape[2]
        else:
            k_cat = k
            v_cat = v
            prefix = 0

        mask = self._causal_prefix_mask(q.shape[2], k_cat.shape[2], prefix, q.device, q.dtype)
        out = F.scaled_dot_product_attention(q, k_cat, v_cat, attn_mask=mask, is_causal=False)

        keep = min(cfg.local_window, k_cat.shape[2])
        new_k = k_cat[:, :, -keep:, :]
        new_v = v_cat[:, :, -keep:, :]
        if cfg.detach_state:
            new_k = new_k.detach()
            new_v = new_v.detach()
        return out, new_k.contiguous(), new_v.contiguous()

    def _retrieval_attention(
        self,
        q: torch.Tensor,
        retrieved_k: Optional[torch.Tensor],
        retrieved_v: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.cfg
        if (not cfg.use_retrieval) or retrieved_k is None or retrieved_v is None:
            return torch.zeros_like(q)
        scale = self.d_head**-0.5
        if retrieved_k.dim() == 4:
            logits = torch.einsum("bhtd,bhrd->bhtr", q, retrieved_k.to(dtype=q.dtype)) * scale
            weights = torch.sigmoid(2.0 * logits)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)
            return torch.einsum("bhtr,bhrd->bhtd", weights, retrieved_v.to(dtype=q.dtype))
        if retrieved_k.dim() == 5:
            logits = torch.einsum("bhtd,bhtrd->bhtr", q, retrieved_k.to(dtype=q.dtype)) * scale
            weights = torch.sigmoid(2.0 * logits)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)
            return torch.einsum("bhtr,bhtrd->bhtd", weights, retrieved_v.to(dtype=q.dtype))
        raise ValueError("retrieved_k/v must be [B,H,R,d] or [B,H,T,R,d]")

    def _slot_write(
        self, k: torch.Tensor, v: torch.Tensor, state: MHDSRA2State, read_mass: torch.Tensor
    ) -> MHDSRA2State:
        cfg = self.cfg
        slot_k = state.slot_k.to(dtype=k.dtype)
        slot_v = state.slot_v.to(dtype=v.dtype)
        kn = F.normalize(k, dim=-1)
        sk = F.normalize(slot_k, dim=-1)
        sim = torch.einsum("bhtd,bhkd->bhtk", kn, sk)
        max_sim = sim.max(dim=-1).values
        novelty = (1.0 - max_sim).clamp(0.0, 1.0)

        tau = self.log_tau_write.exp().to(dtype=k.dtype)
        write_logits = sim * tau
        write_logits = write_logits - cfg.usage_prior * torch.log1p(state.usage).to(
            dtype=k.dtype
        ).unsqueeze(2)
        write_logits = write_logits + cfg.age_write_bias * torch.log1p(state.age).to(
            dtype=k.dtype
        ).unsqueeze(2)

        w_top = min(cfg.write_topk, cfg.slots)
        top_logits, top_idx = torch.topk(write_logits, w_top, dim=-1)
        route = F.softmax(top_logits, dim=-1)
        token_gate = torch.sigmoid(self.token_write_gate(k)).squeeze(-1) * novelty
        weights = route * token_gate.unsqueeze(-1)

        agg_k, mass = self._scatter_values(top_idx, weights, k, cfg.slots)
        agg_v, _ = self._scatter_values(top_idx, weights, v, cfg.slots)
        mass_safe = mass.clamp_min(cfg.eps).to(dtype=k.dtype)
        new_k = agg_k / mass_safe
        new_v = agg_v / mass_safe
        has_write = (mass > cfg.eps).to(dtype=k.dtype)

        conflict = (
            (1.0 - F.cosine_similarity(new_k, slot_k, dim=-1, eps=cfg.eps))
            .clamp(0.0, 2.0)
            .unsqueeze(-1)
        )
        write_gate = (1.0 - torch.exp(-cfg.eta * mass)).to(dtype=k.dtype).clamp(
            0.0, cfg.max_update
        ) * has_write

        age_term = cfg.forget_age * torch.log1p(state.age).unsqueeze(-1).to(dtype=k.dtype)
        forget = cfg.forget_base + age_term + cfg.forget_conflict * write_gate * conflict
        forget = forget.clamp(0.0, 0.95)

        slot_k_next = (1.0 - forget) * slot_k + write_gate * new_k
        slot_k_next = F.normalize(slot_k_next, dim=-1)
        slot_v_next = (1.0 - forget) * slot_v + write_gate * new_v

        wg32 = write_gate.squeeze(-1).to(dtype=torch.float32)
        fg32 = forget.squeeze(-1).to(dtype=torch.float32)
        age_next = (state.age + k.shape[2]).to(dtype=torch.float32)
        age_next = age_next * (1.0 - wg32).clamp(0.0, 1.0)
        usage_next = cfg.usage_decay * state.usage + read_mass + mass.squeeze(-1).to(
            dtype=torch.float32
        )
        conf_new = (1.0 - conflict.squeeze(-1).clamp(0.0, 1.0)).to(dtype=torch.float32)
        conf_next = cfg.conf_decay * state.confidence * (1.0 - fg32) + wg32 * conf_new
        conf_next = conf_next.clamp(0.0, 1.0)

        if cfg.detach_state:
            slot_k_next = slot_k_next.detach()
            slot_v_next = slot_v_next.detach()
            age_next = age_next.detach()
            usage_next = usage_next.detach()
            conf_next = conf_next.detach()

        return MHDSRA2State(
            slot_k=slot_k_next.contiguous(),
            slot_v=slot_v_next.contiguous(),
            age=age_next.contiguous(),
            usage=usage_next.contiguous(),
            confidence=conf_next.contiguous(),
            local_k=state.local_k,
            local_v=state.local_v,
            position=state.position + k.shape[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[MHDSRA2State] = None,
        retrieved_k: Optional[torch.Tensor] = None,
        retrieved_v: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        cfg = self.cfg
        b, t, d = x.shape
        if d != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {d}")
        if state is None:
            state = self.init_state(b, device=x.device, dtype=x.dtype)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._to_heads(q)
        k = self._to_heads(k)
        v = self._to_heads(v)

        slot_out, aux_read = self._slot_read(q, state)
        local_out, new_local_k, new_local_v = self._local_attention(q, k, v, state)
        retrieval_out = self._retrieval_attention(q, retrieved_k, retrieved_v)

        gate_logits = self.fuse_gate(q)
        gates = torch.sigmoid(gate_logits)
        gate_mask = torch.ones_like(gates)
        if not cfg.use_local or cfg.local_window <= 0:
            gate_mask[..., 1] = 0.0
        if (not cfg.use_retrieval) or retrieved_k is None or retrieved_v is None:
            gate_mask[..., 2] = 0.0
        gates = gates * gate_mask
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(cfg.eps)

        y_heads = (
            gates[..., 0:1] * slot_out
            + gates[..., 1:2] * local_out
            + gates[..., 2:3] * retrieval_out
        )
        y = self.out_proj(self._from_heads(y_heads))

        next_state = self._slot_write(k, v, state, aux_read["read_mass"])
        next_state.local_k = new_local_k
        next_state.local_v = new_local_v
        next_state.position = state.position + t

        if return_aux:
            aux = {
                "gates_mean": gates.detach().mean(dim=(0, 2)),
                "read_mass": aux_read["read_mass"].detach(),
                "slot_usage": next_state.usage.detach(),
                "slot_confidence": next_state.confidence.detach(),
            }
            return y, next_state, aux
        return y, next_state

    def forward_step(
        self,
        x_t: torch.Tensor,
        S_prev: Optional[MHDSRA2State] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        retrieved_k: Optional[torch.Tensor] = None,
        retrieved_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, MHDSRA2State, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Run one autoregressive decoding step with DSRA-compatible outputs.

        中文说明:
        - 调用方 / Called by: `MHDSRA2CompatChunkLayer.forward_step`,
          `scripts.json_retrieval_test.greedy_generate_answer`,
          `scripts.json_retrieval_test.rollout_generation_logits`
        - 调用对象 / Calls: `init_state`, `forward`
        - 作用 / Purpose: 为 MHDSRA2 提供逐 token 解码接口，兼容现有 DSRA generation 评测口径
        - 变量 / Variables:
          `x_t` 当前步输入 `[B, 1, D]`, `S_prev` 上一步状态,
          `kv_cache` 为兼容旧接口传入的局部缓存 `(local_k, local_v)`,
          `retrieved_k/retrieved_v` 为可选外部检索记忆
        - 接入 / Integration: 现有按 `model.dsra.forward_step(...)` 调用的脚本可直接接入 MHDSRA2
        - 错误处理 / Error handling:
          当输入不是三维或步长不为 `1` 时抛出 `ValueError`；缺失状态时自动初始化
        - 关键词 / Keywords:
          forward_step|generation|autoregressive|decode|streaming|kv_cache|state|mhdsra2|compat|逐步解码
        """
        if x_t.dim() != 3:
            raise ValueError(f"expected x_t rank=3, got shape={tuple(x_t.shape)}")
        if x_t.shape[1] != 1:
            raise ValueError(
                f"forward_step expects one token, got sequence length={x_t.shape[1]}"
            )

        state = S_prev
        if state is None:
            state = self.init_state(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)

        if (
            kv_cache is not None
            and state.local_k is None
            and state.local_v is None
        ):
            cached_k, cached_v = kv_cache
            state.local_k = cached_k
            state.local_v = cached_v

        out_t, next_state = self.forward(
            x_t,
            state=state,
            retrieved_k=retrieved_k,
            retrieved_v=retrieved_v,
            return_aux=False,
        )
        next_kv_cache = (next_state.local_k, next_state.local_v)
        return out_t, next_state, next_kv_cache

    def diversity_loss(self, state: MHDSRA2State) -> torch.Tensor:
        """Optional training regularizer; not needed for inference.

        Penalizes slot collapse within each head without using an O(K^3) inverse.
        Complexity is O(B * H * K^2 * d), so use only during training and small K.
        """
        sk = F.normalize(state.slot_k, dim=-1)
        gram = torch.einsum("bhkd,bhld->bhkl", sk, sk)
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        off = gram - eye.view(1, 1, gram.shape[-1], gram.shape[-1])
        return (off**2).mean()


def estimate_attention_memory_bytes(
    seq_len: int,
    batch_size: int,
    dim: int,
    heads: int,
    chunk_size: int,
    slots: int,
    read_topk: int,
    write_topk: int,
    local_window: int,
    retrieval_tokens: int,
    dtype_bytes: int = 2,
    page_size: int = 1024,
    landmark_dim: Optional[int] = None,
    keep_full_input_output_on_gpu: bool = False,
) -> Dict[str, int]:
    """Estimate GPU memory for a single MHDSRA2 layer attention working set.

    This excludes model parameters and optimizer states. For ultra-long contexts,
    x must be streamed by chunks; if keep_full_input_output_on_gpu=True, the full
    sequence tensor is also counted and can dominate memory.
    """
    d_head = dim // heads
    landmark_dim = landmark_dim or min(64, d_head)
    pages = (seq_len + page_size - 1) // page_size

    mem = {}
    mem["chunk_input"] = batch_size * chunk_size * dim * dtype_bytes
    mem["qkv_chunk"] = 3 * batch_size * heads * chunk_size * d_head * dtype_bytes
    mem["chunk_output"] = batch_size * chunk_size * dim * dtype_bytes
    mem["slots_key_value"] = 2 * batch_size * heads * slots * d_head * dtype_bytes
    mem["slot_metadata_fp32"] = 3 * batch_size * heads * slots * 4
    mem["slot_logits"] = batch_size * heads * chunk_size * slots * dtype_bytes
    mem["topk_read_write"] = batch_size * heads * chunk_size * (read_topk + write_topk) * (
        dtype_bytes + 4
    )
    mem["local_kv_cache"] = 2 * batch_size * heads * local_window * d_head * dtype_bytes
    mem["retrieval_kv"] = 2 * batch_size * heads * retrieval_tokens * d_head * dtype_bytes
    mem["retrieval_logits"] = batch_size * heads * chunk_size * retrieval_tokens * dtype_bytes
    mem["page_landmarks_gpu"] = batch_size * heads * pages * landmark_dim * dtype_bytes
    if keep_full_input_output_on_gpu:
        mem["full_input_output_gpu"] = 2 * batch_size * seq_len * dim * dtype_bytes
    mem["estimated_total"] = sum(mem.values())
    return mem


def format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} TB"
