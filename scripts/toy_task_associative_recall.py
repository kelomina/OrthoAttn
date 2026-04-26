import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.dsra.dsra_layer import DSRA_Chunk_Layer
from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2

class LocalContextTokenModel(nn.Module):
    def __init__(self, vocab_size, dim, chunk_size=256, local_context_size=4, local_context_mode='sum'):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.local_context_size = max(1, int(local_context_size))
        self.local_context_mode = local_context_mode
        self.embedding = nn.Embedding(vocab_size, dim)
        self.local_context_proj = None
        if self.local_context_mode == 'concat':
            self.local_context_proj = nn.Linear(dim * self.local_context_size, dim)
        elif self.local_context_mode not in {'sum', 'none'}:
            raise ValueError(f"Unsupported local_context_mode: {self.local_context_mode}")

    def build_causal_context(self, emb):
        if self.local_context_mode == 'none':
            return emb

        shifts = [emb]
        for offset in range(1, self.local_context_size):
            shifted = torch.zeros_like(emb)
            shifted[:, offset:, :] = emb[:, :-offset, :]
            shifts.append(shifted)
        if self.local_context_mode == 'sum':
            return sum(shifts)
        return self.local_context_proj(torch.cat(shifts, dim=-1))

    def build_step_context(self, raw_history):
        history = list(raw_history)
        current = history[-1]
        if self.local_context_mode == 'none':
            return current

        padded_history = [current]
        for offset in range(1, self.local_context_size):
            hist_index = len(history) - 1 - offset
            if hist_index >= 0:
                padded_history.append(history[hist_index])
            else:
                padded_history.append(torch.zeros_like(current))

        if self.local_context_mode == 'sum':
            return torch.stack(padded_history, dim=0).sum(dim=0)
        return self.local_context_proj(torch.cat(padded_history, dim=-1))


def _forward_chunked_hidden(model, x):
    _, seq_len = x.shape
    emb = model.build_causal_context(model.embedding(x))

    out_list = []
    S_prev = None
    bypass_kv = None
    S_time_prev = None
    chunk_idx = 0
    for i in range(0, seq_len, model.chunk_size):
        chunk = emb[:, i:i + model.chunk_size, :]
        out_chunk, S_prev, bypass_kv, S_time_prev = model.dsra(
            chunk,
            S_prev=S_prev,
            bypass_kv=bypass_kv,
            S_time_prev=S_time_prev,
            chunk_idx=chunk_idx,
        )
        out_list.append(out_chunk)
        chunk_idx += 1

    out = torch.cat(out_list, dim=1)
    return model.norm(out)


class MHDSRA2CompatChunkLayer(nn.Module):
    def __init__(self, dim, K=128, kr=8, local_window=256):
        super().__init__()
        self.layer = MultiHeadDSRA2(
            MHDSRA2Config(
                dim=dim,
                heads=max(1, min(8, dim // 16 if dim >= 16 else 1)),
                slots=K,
                read_topk=max(1, min(kr, K)),
                write_topk=max(1, min(kr, K)),
                local_window=max(1, int(local_window)),
                use_local=True,
                use_retrieval=True,
                detach_state=True,
            )
        )

    def forward(self, chunk, S_prev=None, bypass_kv=None, S_time_prev=None, chunk_idx=None):
        """Expose DSRA-style chunk forward over MHDSRA2.

        中文说明:
        - 调用方 / Called by: `_forward_chunked_hidden`,
          `scripts.json_retrieval_test.greedy_generate_answer`
        - 调用对象 / Calls: `MultiHeadDSRA2.forward`
        - 作用 / Purpose: 把 MHDSRA2 适配为 DSRA 现有四元组接口，避免改重现有评测脚本
        - 变量 / Variables:
          `chunk` 当前 chunk hidden, `S_prev` 为 `MHDSRA2State`,
          `bypass_kv` 为兼容字段，会映射到 `local_k/local_v`
        - 接入 / Integration: 可挂到 `model.dsra` 上复用现有 chunked forward/generation 逻辑
        - 错误处理 / Error handling: 底层张量形状异常直接向上抛出
        - 关键词 / Keywords:
          compat|chunk|forward|mhdsra2|dsra|adapter|state|local_cache|generation|兼容
        """
        if (
            S_prev is not None
            and bypass_kv is not None
            and S_prev.local_k is None
            and S_prev.local_v is None
        ):
            S_prev.local_k, S_prev.local_v = bypass_kv
        out_chunk, next_state = self.layer(chunk, state=S_prev)
        return out_chunk, next_state, (next_state.local_k, next_state.local_v), None

    def forward_step(self, step_input, S_prev=None, kv_cache=None):
        """Expose DSRA-style autoregressive step API over MHDSRA2.

        中文说明:
        - 调用方 / Called by: `scripts.json_retrieval_test.greedy_generate_answer`,
          `scripts.json_retrieval_test.rollout_generation_logits`
        - 调用对象 / Calls: `MultiHeadDSRA2.forward_step`
        - 作用 / Purpose: 让现有 generation 评测无需分支即可调用 MHDSRA2
        - 变量 / Variables:
          `step_input` 当前步 embedding, `S_prev` 为 MHDSRA2 流式状态,
          `kv_cache` 为兼容旧接口传入的局部 KV 缓存
        - 接入 / Integration: 与 `DSRA_Chunk_Layer.forward_step` 保持同签名
        - 错误处理 / Error handling: 依赖底层 `forward_step` 的输入维度校验
        - 关键词 / Keywords:
          forward_step|generation|compat|mhdsra2|decode|autoregressive|kv_cache|adapter|token|兼容
        """
        return self.layer.forward_step(step_input, S_prev=S_prev, kv_cache=kv_cache)


class DSRAModel(LocalContextTokenModel):
    def __init__(
        self,
        vocab_size,
        dim,
        K=128,
        kr=8,
        chunk_size=256,
        pe_mode='none',
        use_orthogonal_update=True,
        use_bypass=True,
        local_context_size=4,
        local_context_mode='sum',
    ):
        super().__init__(
            vocab_size=vocab_size,
            dim=dim,
            chunk_size=chunk_size,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
        )
        self.dsra = DSRA_Chunk_Layer(
            dim,
            K=K,
            kr=kr,
            pe_mode=pe_mode,
            use_orthogonal_update=use_orthogonal_update,
            use_bypass=use_bypass,
        )
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(self, x, return_hidden=False):
        out = _forward_chunked_hidden(self, x)
        logits = self.out_proj(out)

        if return_hidden:
            return logits, out
        return logits


class MHDSRA2Model(LocalContextTokenModel):
    def __init__(
        self,
        vocab_size,
        dim,
        K=128,
        kr=8,
        chunk_size=256,
        local_context_size=4,
        local_context_mode='sum',
    ):
        super().__init__(
            vocab_size=vocab_size,
            dim=dim,
            chunk_size=chunk_size,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
        )
        local_window = max(chunk_size, local_context_size)
        self.dsra = MHDSRA2CompatChunkLayer(dim, K=K, kr=kr, local_window=local_window)
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(self, x, return_hidden=False):
        """Run chunked MHDSRA2 token modeling with DSRA-compatible scaffolding.

        中文说明:
        - 调用方 / Called by: `scripts.json_retrieval_test.evaluate_teacher_forced`,
          `scripts.json_retrieval_test.evaluate_generation`
        - 调用对象 / Calls: `_forward_chunked_hidden`, `nn.Linear`
        - 作用 / Purpose: 在保留 `LocalContextTokenModel` 接口的前提下接入 MHDSRA2 检索任务模型
        - 变量 / Variables:
          `x` token ids, `out` hidden states, `logits` 词表 logits,
          `return_hidden` 控制是否返回隐藏状态
        - 接入 / Integration: 由 `build_retrieval_model(model_type="mhdsra2")` 创建
        - 错误处理 / Error handling: 继承 chunk 前向与线性层的异常抛出行为
        - 关键词 / Keywords:
          mhdsra2|retrieval_model|forward|token_model|chunked|hidden|logits|adapter|benchmark|接入
        """
        out = _forward_chunked_hidden(self, x)
        logits = self.out_proj(out)

        if return_hidden:
            return logits, out
        return logits


class StandardAttentionChunkLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _causal_attention(self, query, key, value, prefix_len=0):
        scale = 1.0 / math.sqrt(self.dim)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        query_len = query.shape[1]
        key_len = key.shape[1]
        query_positions = torch.arange(query_len, device=query.device).unsqueeze(-1)
        key_positions = torch.arange(key_len, device=key.device).unsqueeze(0)
        causal_mask = key_positions > (prefix_len + query_positions)
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, value)

    def forward(self, chunk, S_prev=None, bypass_kv=None, S_time_prev=None, chunk_idx=None):
        cached_k = None
        cached_v = None
        if bypass_kv is not None:
            cached_k, cached_v = bypass_kv

        query = self.q_proj(chunk)
        new_k = self.k_proj(chunk)
        new_v = self.v_proj(chunk)
        if cached_k is None:
            full_k = new_k
            full_v = new_v
            prefix_len = 0
        else:
            full_k = torch.cat([cached_k, new_k], dim=1)
            full_v = torch.cat([cached_v, new_v], dim=1)
            prefix_len = cached_k.shape[1]

        attended = self._causal_attention(query, full_k, full_v, prefix_len=prefix_len)
        return self.out_proj(attended), None, (full_k, full_v), None

    def forward_step(self, step_input, S_prev=None, kv_cache=None):
        cached_k = None
        cached_v = None
        if kv_cache is not None:
            cached_k, cached_v = kv_cache

        query = self.q_proj(step_input)
        new_k = self.k_proj(step_input)
        new_v = self.v_proj(step_input)
        if cached_k is None:
            full_k = new_k
            full_v = new_v
        else:
            full_k = torch.cat([cached_k, new_k], dim=1)
            full_v = torch.cat([cached_v, new_v], dim=1)

        attended = F.scaled_dot_product_attention(query, full_k, full_v, is_causal=False)
        return self.out_proj(attended), None, (full_k, full_v)


class StandardAttentionModel(LocalContextTokenModel):
    def __init__(
        self,
        vocab_size,
        dim,
        chunk_size=256,
        local_context_size=4,
        local_context_mode='sum',
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            dim=dim,
            chunk_size=chunk_size,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
        )
        self.dsra = StandardAttentionChunkLayer(dim)
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(self, x, return_hidden=False):
        out = _forward_chunked_hidden(self, x)
        logits = self.out_proj(out)

        if return_hidden:
            return logits, out
        return logits


class SlidingWindowAttentionChunkLayer(nn.Module):
    def __init__(self, dim, window_size=1024):
        super().__init__()
        self.dim = dim
        self.window_size = max(1, int(window_size))
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _causal_attention(self, query, key, value, prefix_len=0):
        scale = 1.0 / math.sqrt(self.dim)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        query_len = query.shape[1]
        key_len = key.shape[1]
        query_positions = torch.arange(query_len, device=query.device).unsqueeze(-1)
        key_positions = torch.arange(key_len, device=key.device).unsqueeze(0)
        causal_mask = key_positions > (prefix_len + query_positions)
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, value)

    def _truncate_cache(self, key, value):
        if key.shape[1] <= self.window_size:
            return key, value
        return key[:, -self.window_size:, :], value[:, -self.window_size:, :]

    def forward(self, chunk, S_prev=None, bypass_kv=None, S_time_prev=None, chunk_idx=None):
        cached_k = None
        cached_v = None
        if bypass_kv is not None:
            cached_k, cached_v = bypass_kv

        query = self.q_proj(chunk)
        new_k = self.k_proj(chunk)
        new_v = self.v_proj(chunk)
        if cached_k is None:
            full_k = new_k
            full_v = new_v
            prefix_len = 0
        else:
            full_k = torch.cat([cached_k, new_k], dim=1)
            full_v = torch.cat([cached_v, new_v], dim=1)
            prefix_len = cached_k.shape[1]

        attended = self._causal_attention(query, full_k, full_v, prefix_len=prefix_len)
        next_cache = self._truncate_cache(full_k, full_v)
        return self.out_proj(attended), None, next_cache, None

    def forward_step(self, step_input, S_prev=None, kv_cache=None):
        cached_k = None
        cached_v = None
        if kv_cache is not None:
            cached_k, cached_v = kv_cache

        query = self.q_proj(step_input)
        new_k = self.k_proj(step_input)
        new_v = self.v_proj(step_input)
        if cached_k is None:
            full_k = new_k
            full_v = new_v
        else:
            full_k = torch.cat([cached_k, new_k], dim=1)
            full_v = torch.cat([cached_v, new_v], dim=1)

        attended = F.scaled_dot_product_attention(query, full_k, full_v, is_causal=False)
        next_cache = self._truncate_cache(full_k, full_v)
        return self.out_proj(attended), None, next_cache


class SlidingWindowAttentionModel(LocalContextTokenModel):
    def __init__(
        self,
        vocab_size,
        dim,
        chunk_size=256,
        local_context_size=4,
        local_context_mode='sum',
        window_size=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            dim=dim,
            chunk_size=chunk_size,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
        )
        self.window_size = max(chunk_size * 4, int(window_size or chunk_size * 4))
        self.dsra = SlidingWindowAttentionChunkLayer(dim, window_size=self.window_size)
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(self, x, return_hidden=False):
        out = _forward_chunked_hidden(self, x)
        logits = self.out_proj(out)

        if return_hidden:
            return logits, out
        return logits


class SparseAttentionChunkLayer(nn.Module):
    def __init__(self, dim, local_window=512, global_stride=1024):
        super().__init__()
        self.dim = dim
        self.local_window = max(1, int(local_window))
        self.global_stride = max(1, int(global_stride))
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _build_sparse_cache(self, key, value, positions, total_seen):
        local_start = max(0, total_seen - self.local_window)
        keep_mask = positions >= local_start
        keep_mask = keep_mask | (positions % self.global_stride == 0)
        keep_indices = keep_mask.nonzero(as_tuple=False).squeeze(-1)
        if keep_indices.numel() == 0:
            keep_indices = positions.new_tensor([positions.shape[0] - 1])
        return (
            key[:, keep_indices, :],
            value[:, keep_indices, :],
            positions[keep_indices],
            total_seen,
        )

    def _attend(self, query, query_positions, key, value, key_positions):
        scale = 1.0 / math.sqrt(self.dim)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(-1)
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, value)

    def forward(self, chunk, S_prev=None, bypass_kv=None, S_time_prev=None, chunk_idx=None):
        if bypass_kv is None:
            cached_k = None
            cached_v = None
            cached_positions = None
            total_seen = 0
        else:
            cached_k, cached_v, cached_positions, total_seen = bypass_kv

        query = self.q_proj(chunk)
        new_k = self.k_proj(chunk)
        new_v = self.v_proj(chunk)
        new_positions = torch.arange(
            total_seen,
            total_seen + chunk.shape[1],
            device=chunk.device,
            dtype=torch.long,
        )
        if cached_k is None:
            full_k = new_k
            full_v = new_v
            full_positions = new_positions
        else:
            full_k = torch.cat([cached_k, new_k], dim=1)
            full_v = torch.cat([cached_v, new_v], dim=1)
            full_positions = torch.cat([cached_positions, new_positions], dim=0)

        attended = self._attend(query, new_positions, full_k, full_v, full_positions)
        next_cache = self._build_sparse_cache(
            full_k,
            full_v,
            full_positions,
            total_seen + chunk.shape[1],
        )
        return self.out_proj(attended), None, next_cache, None

    def forward_step(self, step_input, S_prev=None, kv_cache=None):
        if kv_cache is None:
            cached_k = None
            cached_v = None
            cached_positions = None
            total_seen = 0
        else:
            cached_k, cached_v, cached_positions, total_seen = kv_cache

        query = self.q_proj(step_input)
        new_k = self.k_proj(step_input)
        new_v = self.v_proj(step_input)
        new_position = torch.tensor([total_seen], device=step_input.device, dtype=torch.long)
        if cached_k is None:
            full_k = new_k
            full_v = new_v
            full_positions = new_position
        else:
            full_k = torch.cat([cached_k, new_k], dim=1)
            full_v = torch.cat([cached_v, new_v], dim=1)
            full_positions = torch.cat([cached_positions, new_position], dim=0)

        attended = self._attend(query, new_position, full_k, full_v, full_positions)
        next_cache = self._build_sparse_cache(
            full_k,
            full_v,
            full_positions,
            total_seen + 1,
        )
        return self.out_proj(attended), None, next_cache


class SparseAttentionModel(LocalContextTokenModel):
    def __init__(
        self,
        vocab_size,
        dim,
        chunk_size=256,
        local_context_size=4,
        local_context_mode='sum',
        sparse_local_window=None,
        sparse_global_stride=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            dim=dim,
            chunk_size=chunk_size,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
        )
        self.sparse_local_window = max(chunk_size * 2, int(sparse_local_window or chunk_size * 2))
        self.sparse_global_stride = max(chunk_size * 4, int(sparse_global_stride or chunk_size * 4))
        self.dsra = SparseAttentionChunkLayer(
            dim,
            local_window=self.sparse_local_window,
            global_stride=self.sparse_global_stride,
        )
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(self, x, return_hidden=False):
        out = _forward_chunked_hidden(self, x)
        logits = self.out_proj(out)

        if return_hidden:
            return logits, out
        return logits


class LinearAttentionChunkLayer(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _feature_map(self, x):
        return F.elu(x) + 1.0

    def _init_state(self, batch_size, device, dtype):
        key_sum = torch.zeros(batch_size, self.dim, device=device, dtype=dtype)
        kv_sum = torch.zeros(batch_size, self.dim, self.dim, device=device, dtype=dtype)
        return key_sum, kv_sum

    def forward(self, chunk, S_prev=None, bypass_kv=None, S_time_prev=None, chunk_idx=None):
        query = self._feature_map(self.q_proj(chunk))
        key = self._feature_map(self.k_proj(chunk))
        value = self.v_proj(chunk)

        if S_prev is None:
            key_sum, kv_sum = self._init_state(chunk.shape[0], chunk.device, chunk.dtype)
        else:
            key_sum, kv_sum = S_prev

        outputs = []
        for step_idx in range(chunk.shape[1]):
            key_t = key[:, step_idx, :]
            value_t = value[:, step_idx, :]
            key_sum = key_sum + key_t
            kv_sum = kv_sum + torch.einsum('bd,be->bde', key_t, value_t)
            query_t = query[:, step_idx, :]
            numerator = torch.einsum('bd,bde->be', query_t, kv_sum)
            denominator = torch.einsum('bd,bd->b', query_t, key_sum).unsqueeze(-1) + self.eps
            outputs.append((numerator / denominator).unsqueeze(1))

        out = self.out_proj(torch.cat(outputs, dim=1))
        return out, (key_sum, kv_sum), None, None

    def forward_step(self, step_input, S_prev=None, kv_cache=None):
        query = self._feature_map(self.q_proj(step_input))
        key = self._feature_map(self.k_proj(step_input))
        value = self.v_proj(step_input)

        if S_prev is None:
            key_sum, kv_sum = self._init_state(step_input.shape[0], step_input.device, step_input.dtype)
        else:
            key_sum, kv_sum = S_prev

        key_t = key[:, 0, :]
        value_t = value[:, 0, :]
        key_sum = key_sum + key_t
        kv_sum = kv_sum + torch.einsum('bd,be->bde', key_t, value_t)
        query_t = query[:, 0, :]
        numerator = torch.einsum('bd,bde->be', query_t, kv_sum)
        denominator = torch.einsum('bd,bd->b', query_t, key_sum).unsqueeze(-1) + self.eps
        out = self.out_proj((numerator / denominator).unsqueeze(1))
        return out, (key_sum, kv_sum), None


class LinearAttentionModel(LocalContextTokenModel):
    def __init__(
        self,
        vocab_size,
        dim,
        chunk_size=256,
        local_context_size=4,
        local_context_mode='sum',
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            dim=dim,
            chunk_size=chunk_size,
            local_context_size=local_context_size,
            local_context_mode=local_context_mode,
        )
        self.dsra = LinearAttentionChunkLayer(dim)
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(self, x, return_hidden=False):
        out = _forward_chunked_hidden(self, x)
        logits = self.out_proj(out)

        if return_hidden:
            return logits, out
        return logits


def build_fixed_associative_mapping(vocab_size, num_pairs, seed=0, noise_pool_size=None):
    required_tokens = 2 * num_pairs
    if vocab_size < required_tokens + 5:
        raise ValueError("Vocabulary too small for a fixed associative mapping.")

    rng = random.Random(seed)
    token_pool = rng.sample(list(range(4, vocab_size)), vocab_size - 4)
    keys = token_pool[:num_pairs]
    values = token_pool[num_pairs:2 * num_pairs]
    noise_tokens = token_pool[2 * num_pairs:]

    if noise_pool_size is not None:
        noise_tokens = noise_tokens[:noise_pool_size]
    if not noise_tokens:
        raise ValueError("Fixed mapping must leave at least one noise token.")

    return {
        "pairs": list(zip(keys, values)),
        "noise_tokens": noise_tokens,
    }


def generate_associative_recall_data(
    batch_size,
    seq_len,
    vocab_size,
    num_pairs=10,
    fixed_pairs=None,
    fixed_noise_tokens=None,
):
    X = torch.zeros(batch_size, seq_len, dtype=torch.long)
    Y = torch.zeros(batch_size, seq_len, dtype=torch.long)
    query_token = 1
    key_token = 2
    value_token = 3
    required_tokens = (2 * num_pairs) + 1

    if vocab_size < required_tokens + 4:
        raise ValueError("Vocabulary too small for disjoint keys, values, and noise.")

    for b in range(batch_size):
        if fixed_pairs is None:
            available_tokens = random.sample(range(4, vocab_size), required_tokens)
            keys = available_tokens[:num_pairs]
            values = available_tokens[num_pairs:2 * num_pairs]
            noise_tokens = available_tokens[2 * num_pairs:]
            pairs = list(zip(keys, values))
        else:
            pairs = list(fixed_pairs)
            if len(pairs) != num_pairs:
                raise ValueError("fixed_pairs size must match num_pairs.")
            if fixed_noise_tokens is None:
                used_tokens = {token for pair in pairs for token in pair}
                noise_tokens = [token for token in range(4, vocab_size) if token not in used_tokens]
            else:
                noise_tokens = list(fixed_noise_tokens)
            if not noise_tokens:
                raise ValueError("Fixed mapping must leave at least one noise token.")

        for i in range(seq_len):
            X[b, i] = random.choice(noise_tokens)

        tail_start = seq_len - 2 - (num_pairs * 4)
        available_positions = list(range(tail_start, seq_len - 2, 4))
        if len(available_positions) < num_pairs:
            raise ValueError("Sequence too short to fit the required number of pairs.")

        positions = sorted(random.sample(available_positions, num_pairs))
        for (k, v), pos in zip(pairs, positions):
            X[b, pos] = key_token
            X[b, pos + 1] = k
            X[b, pos + 2] = value_token
            X[b, pos + 3] = v

        query_key, target_value = random.choice(pairs)
        query_pos = seq_len - 2
        X[b, query_pos] = query_token
        X[b, query_pos + 1] = query_key
        Y[b, query_pos + 1] = target_value

    return X, Y

def train_step(model, X, Y, optimizer, criterion):
    optimizer.zero_grad()
    logits = model(X)
    B, SeqLen = X.shape
    target_indices = (Y != 0).nonzero(as_tuple=True)

    if len(target_indices[0]) != B:
        logits_target = logits[:, -1, :]
        targets = Y[:, -1]
    else:
        logits_target = logits[target_indices[0], target_indices[1], :]
        targets = Y[target_indices[0], target_indices[1]]

    loss = criterion(logits_target, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    preds = logits_target.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    acc = correct / B

    return loss.item(), acc

def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    vocab_size = 100
    dim = 64
    seq_len = 1024
    chunk_size = 256
    batch_size = 16
    epochs = 1500

    model = DSRAModel(vocab_size=vocab_size, dim=dim, K=64, kr=8, chunk_size=chunk_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for epoch in range(epochs):
        X, Y = generate_associative_recall_data(batch_size, seq_len, vocab_size, num_pairs=20)
        X, Y = X.to(device), Y.to(device)

        loss_val, acc = train_step(model, X, Y, optimizer, criterion)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss_val:.4f} | Accuracy: {acc*100:.1f}%")

            if acc == 1.0 and loss_val < 0.05:
                print("Task solved successfully!")
                break

if __name__ == "__main__":
    train()
