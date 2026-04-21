import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rotary_pos_emb(x, offset=0):
    B, T, D = x.shape
    assert D % 2 == 0, "Dimension must be even for RoPE"
    positions = torch.arange(offset, offset + T, dtype=torch.float32, device=x.device).unsqueeze(1)
    freqs = torch.exp(torch.arange(0, D, 2, dtype=torch.float32, device=x.device) * -(torch.log(torch.tensor(10000.0)) / D))
    angles = positions * freqs
    sin = torch.sin(angles).unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    x_rot = torch.zeros_like(x)
    x_rot[..., 0::2] = x1 * cos - x2 * sin
    x_rot[..., 1::2] = x1 * sin + x2 * cos
    return x_rot

def get_alibi_mask(seq_len_q, seq_len_k, is_causal=True, device=None, dtype=None):
    m = 0.125
    q_idx = torch.arange(seq_len_q, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_len_k, device=device).unsqueeze(0)
    dist = k_idx - q_idx
    mask = dist.to(dtype) * m
    if is_causal:
        causal_mask = (k_idx > q_idx)
        mask = mask.masked_fill(causal_mask, float('-inf'))
    # Change shape to [1, 1, seq_len_q, seq_len_k] to broadcast correctly
    # Because F.scaled_dot_product_attention expects attn_mask to be broadcastable to [B, num_heads, seq_len_q, seq_len_k]
    # Since we have no num_heads explicitly in this implementation, Q is [B, T, D] which it treats as 1 head.
    return mask.unsqueeze(0).unsqueeze(0)

def get_chunk_causal_mask(seq_len_q, seq_len_k, prefix_len=0, device=None, dtype=None):
    """创建分块因果注意力掩码，允许访问前缀分块但限制未来分块。"""
    q_idx = torch.arange(seq_len_q, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_len_k, device=device).unsqueeze(0)
    allowed_k = prefix_len + q_idx
    mask = torch.zeros(seq_len_q, seq_len_k, device=device, dtype=dtype)
    mask = mask.masked_fill(k_idx > allowed_k, float('-inf'))
    return mask.unsqueeze(0)

class DSRA_Chunk_Layer(nn.Module):
    def __init__(self, dim, K=512, kr=16, eta=0.1, decay_lambda=0.01, use_orthogonal_update=True, use_bypass=True, pe_mode='none'):
        super().__init__()
        self.dim = dim
        self.K = K
        self.kr = kr
        self.eta = eta
        self.decay_lambda = decay_lambda
        self.use_orthogonal_update = use_orthogonal_update
        self.use_bypass = use_bypass
        self.pe_mode = pe_mode
        self.time_decay_alpha = 0.01

        
        # S_init: [K, dim]
        self.S_init = nn.Parameter(torch.randn(K, dim) / (dim ** 0.5))
        
        self.W_q = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_n = nn.Linear(dim + 1, K) 
        self.W_m = nn.Linear(dim, 1)

    def sparse_topk_distribution(self, logits):
        topk = min(self.kr, logits.size(-1))
        topk_logits, topk_idx = torch.topk(logits, topk, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)
        return torch.zeros_like(logits).scatter(-1, topk_idx, topk_probs)

    def forward(self, x, S_prev=None, bypass_kv=None, S_time_prev=None, chunk_idx=0):
        """
        x: [B, T, D]
        S_prev: [B, K, D]
        S_time_prev: [B, K]
        """
        B, T, D = x.shape
        chunk_time_start = chunk_idx * T
        
        if S_prev is None:
            S_prev = self.S_init.unsqueeze(0).expand(B, -1, -1)
            
        if self.pe_mode == 'timestamps' and S_time_prev is None:
            S_time_prev = torch.zeros(B, self.K, device=x.device, dtype=torch.float32)
            
        Q = self.W_q(x)
        V = self.W_v(x)
        Q_read = F.normalize(Q, dim=-1)
        S_read = F.normalize(S_prev, dim=-1)
        read_logits = torch.einsum('btd,bkd->btk', Q_read, S_read) * (self.dim ** 0.5)
        
        if self.pe_mode == 'timestamps':
            t_curr = chunk_time_start + torch.arange(T, device=x.device, dtype=torch.float32)
            # t_curr: [T] -> [1, T, 1]
            t_curr = t_curr.unsqueeze(0).unsqueeze(-1)
            # S_time_prev: [B, K] -> [B, 1, K]
            s_time = S_time_prev.unsqueeze(1)
            
            # Resulting time_diff shape: [B, T, K]
            time_diff = t_curr - s_time
            # time_diff should be non-negative. If S_time_prev is 0, difference is large.
            # We apply penalty to older slots
            
            read_logits = read_logits - self.time_decay_alpha * time_diff

        
        r_read = self.sparse_topk_distribution(read_logits)
        context = torch.einsum('btk,bkd->btd', r_read, S_prev)
        
        # 2. Instruction Bypass (Local Window)
        if self.use_bypass:
            instr_prob = torch.sigmoid(self.W_m(Q)) # [B, T, 1]
        else:
            instr_prob = torch.zeros(B, T, 1, device=x.device)
            
        Q_bypass = Q
        if self.pe_mode == 'rope':
            Q_bypass = apply_rotary_pos_emb(Q, offset=chunk_time_start)

        if bypass_kv is not None:
            K_bypass, V_bypass = bypass_kv
            prev_len = K_bypass.shape[1]
            if self.pe_mode == 'rope':
                K_bypass = apply_rotary_pos_emb(K_bypass, offset=max(chunk_time_start - prev_len, 0))

            if self.pe_mode == 'rope':
                K_current = Q_bypass
            else:
                K_current = Q

            K_cat = torch.cat([K_bypass, K_current], dim=1)
            V_cat = torch.cat([V_bypass, V], dim=1)
            attn_mask = None
            if self.pe_mode == 'alibi':
                alibi_mask = get_alibi_mask(T, prev_len + T, is_causal=False, device=x.device, dtype=x.dtype).squeeze(0)
                causal_mask = get_chunk_causal_mask(T, prev_len + T, prefix_len=prev_len, device=x.device, dtype=x.dtype)
                attn_mask = alibi_mask + causal_mask
            else:
                attn_mask = get_chunk_causal_mask(T, prev_len + T, prefix_len=prev_len, device=x.device, dtype=x.dtype)

            bypass_out = F.scaled_dot_product_attention(Q_bypass, K_cat, V_cat, is_causal=False, attn_mask=attn_mask)
        else:
            if self.pe_mode == 'rope':
                K_bypass_self = Q_bypass
            else:
                K_bypass_self = Q

            attn_mask = None
            if self.pe_mode == 'alibi':
                attn_mask = get_alibi_mask(T, T, is_causal=True, device=x.device, dtype=x.dtype)
                attn_mask = attn_mask.squeeze(0)

            bypass_out = F.scaled_dot_product_attention(Q_bypass, K_bypass_self, V, is_causal=(attn_mask is None), attn_mask=attn_mask)
            
        out = context + instr_prob * bypass_out
        
        sim = F.cosine_similarity(V.unsqueeze(2), S_prev.unsqueeze(1), dim=-1)
        novelty = 1.0 - sim.max(dim=-1)[0]
        write_logits = self.W_n(torch.cat([novelty.unsqueeze(-1), Q], dim=-1)) + read_logits
        r_write = self.sparse_topk_distribution(write_logits)
        w = (1 - self.decay_lambda) ** torch.arange(T - 1, -1, -1, device=x.device, dtype=x.dtype)
        weighted_write = r_write * novelty.unsqueeze(-1).clamp(0.0, 1.0) * w.view(1, T, 1)
        slot_mass = weighted_write.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        V_agg = torch.einsum('btk,btd->bkd', weighted_write, V) / slot_mass

        if self.use_orthogonal_update:
            S_cov = torch.bmm(S_prev, S_prev.transpose(1, 2)) + 1e-5 * torch.eye(self.K, device=x.device)
            S_cov_inv = torch.inverse(S_cov)
            SV = torch.bmm(V_agg, S_prev.transpose(1, 2))
            proj_coeff = torch.bmm(SV, S_cov_inv)
            V_proj = torch.bmm(proj_coeff, S_prev)
            V_orth = V_agg - V_proj
        else:
            V_orth = V_agg

        self.last_V_orth = V_orth
        S_next = (1 - self.decay_lambda) * S_prev + self.eta * V_orth

        S_time_next = S_time_prev
        if self.pe_mode == 'timestamps':
            max_write_prob = weighted_write.max(dim=1)[0]
            S_time_next = S_time_prev * (1 - max_write_prob) + (chunk_time_start + T) * max_write_prob
            
        return out, S_next, (Q, V), S_time_next

    def forward_step(self, x_t, S_prev, kv_cache=None):
        """
        Autoregressive step generation interface.
        x_t: [B, 1, D] (Current token embedding)
        S_prev: [B, K, D] (Accumulated state)
        kv_cache: (K_cache, V_cache) from the current decoding window
                  where K_cache, V_cache are [B, window_len, D]
        
        Returns: out_t [B, 1, D], S_next [B, K, D], kv_cache_next
        """
        B, _, D = x_t.shape
        
        if S_prev is None:
            S_prev = self.S_init.unsqueeze(0).expand(B, -1, -1)
            
        q_t = self.W_q(x_t)
        v_t = self.W_v(x_t)
        q_read = F.normalize(q_t, dim=-1)
        s_read = F.normalize(S_prev, dim=-1)
        read_logits = torch.einsum('btd,bkd->btk', q_read, s_read) * (self.dim ** 0.5)
        r_read = self.sparse_topk_distribution(read_logits)
        context_t = torch.einsum('btk,bkd->btd', r_read, S_prev)
        
        # 2. Bypass
        if self.use_bypass:
            instr_prob = torch.sigmoid(self.W_m(q_t))
        else:
            instr_prob = torch.zeros(B, 1, 1, device=x_t.device)
            
        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            K_cache = torch.cat([K_cache, q_t], dim=1)
            V_cache = torch.cat([V_cache, v_t], dim=1)
        else:
            K_cache, V_cache = q_t, v_t

        bypass_out_t = F.scaled_dot_product_attention(q_t, K_cache, V_cache, is_causal=False)
        out_t = context_t + instr_prob * bypass_out_t

        sim = F.cosine_similarity(v_t.unsqueeze(2), S_prev.unsqueeze(1), dim=-1)
        novelty = 1.0 - sim.max(dim=-1)[0]
        write_logits = self.W_n(torch.cat([novelty.unsqueeze(-1), q_t], dim=-1)) + read_logits
        r_write = self.sparse_topk_distribution(write_logits)
        weighted_write = r_write * novelty.unsqueeze(-1).clamp(0.0, 1.0)
        slot_mass = weighted_write.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        V_agg = torch.einsum('btk,btd->bkd', weighted_write, v_t) / slot_mass

        if self.use_orthogonal_update:
            S_cov = torch.bmm(S_prev, S_prev.transpose(1, 2)) + 1e-5 * torch.eye(self.K, device=x_t.device)
            S_cov_inv = torch.inverse(S_cov)
            SV = torch.bmm(V_agg, S_prev.transpose(1, 2))
            proj_coeff = torch.bmm(SV, S_cov_inv)
            V_proj = torch.bmm(proj_coeff, S_prev)
            V_orth = V_agg - V_proj
        else:
            V_orth = V_agg
            
        S_next = (1 - self.decay_lambda) * S_prev + self.eta * V_orth

        return out_t, S_next, (K_cache, V_cache)
