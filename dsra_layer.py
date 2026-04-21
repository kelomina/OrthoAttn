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
            
        Q = self.W_q(x) # [B, T, D]
        V = self.W_v(x) # [B, T, D]
        
        # 1. Read Routing
        # [B, T, D] @ [B, D, K] -> [B, T, K]
        read_logits = torch.einsum('btd,bkd->btk', Q, S_prev) / (self.dim ** 0.5)
        
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

        
        # Gumbel Softmax for differentiable TopK
        # For training stability, we use softmax over topk
        topk_logits, topk_idx = torch.topk(read_logits, self.kr, dim=-1) # [B, T, kr]
        topk_probs = F.softmax(topk_logits, dim=-1) # [B, T, kr]
        
        # Reconstruct full sparse distribution
        # Need to ensure scatter operates on correct dimension.
        # read_logits is [B, T, K], topk_idx is [B, T, kr], topk_probs is [B, T, kr]
        r_read = torch.zeros_like(read_logits).scatter(-1, topk_idx, topk_probs) # [B, T, K]
        
        # Read from state
        # [B, T, K] @ [B, K, D] -> [B, T, D]
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
            if self.pe_mode == 'rope':
                # K_bypass is from previous chunk
                K_bypass = apply_rotary_pos_emb(K_bypass, offset=chunk_time_start - T)
                
            attn_mask = None
            if self.pe_mode == 'alibi':
                # ALiBi expects a shape broadcastable to [B, num_heads, seq_len_q, seq_len_k].
                # Since we don't use multi-head explicitly, Q is [B, T, D], we need [B, 1, T, T] or [1, 1, T, T]
                attn_mask = get_alibi_mask(T, T, is_causal=False, device=x.device, dtype=x.dtype)
                # But scaled_dot_product_attention for [B, T, D] expects mask [B, T, T] or [1, T, T]
                attn_mask = attn_mask.squeeze(0) # [1, T, T]
                
            # bypass_kv is from the previous chunk, so all its tokens are in the past
            # Hence, no causal mask is needed for cross-attention to the previous chunk
            bypass_out = F.scaled_dot_product_attention(Q_bypass, K_bypass, V_bypass, is_causal=(attn_mask is None), attn_mask=attn_mask)
        else:
            if self.pe_mode == 'rope':
                K_bypass_self = Q_bypass
            else:
                K_bypass_self = Q
                
            attn_mask = None
            if self.pe_mode == 'alibi':
                attn_mask = get_alibi_mask(T, T, is_causal=True, device=x.device, dtype=x.dtype)
                attn_mask = attn_mask.squeeze(0) # [1, T, T]
                
            # Self-attention within chunk requires causal mask
            bypass_out = F.scaled_dot_product_attention(Q_bypass, K_bypass_self, V, is_causal=(attn_mask is None), attn_mask=attn_mask)
            
        out = (1 - instr_prob) * context + instr_prob * bypass_out
        
        # 3. Orthogonal Write Routing & Update
        # Novelty detection
        sim = F.cosine_similarity(V.unsqueeze(2), S_prev.unsqueeze(1), dim=-1) # [B, T, K]
        novelty = 1.0 - sim.max(dim=-1)[0] # [B, T]
        
        # Write routing logits
        write_logits = self.W_n(torch.cat([novelty.unsqueeze(-1), Q], dim=-1)) # [B, T, K]
        r_write = torch.sigmoid(write_logits) # [B, T, K]
        
        # Aggregate chunk information to write
        # Generate time decay weights for aggregation
        w = (1 - self.decay_lambda) ** torch.arange(T - 1, -1, -1, device=x.device, dtype=x.dtype)
        w = w / w.sum() # Normalize
        
        # [B, T, K] @ [B, T, D] with weights [T]
        V_agg = torch.einsum('btk,btd,t->bkd', r_write, V, w)
        
        # Orthogonal Projection: P = S^T (S S^T)^-1 S
        if self.use_orthogonal_update:
            # We project V_agg to be orthogonal to the row space of S_prev
            S_cov = torch.bmm(S_prev, S_prev.transpose(1, 2)) + 1e-5 * torch.eye(self.K, device=x.device)
            S_cov_inv = torch.inverse(S_cov)
            
            SV = torch.bmm(V_agg, S_prev.transpose(1, 2)) # [B, K, K]
            proj_coeff = torch.bmm(SV, S_cov_inv) # [B, K, K]
            V_proj = torch.bmm(proj_coeff, S_prev) # [B, K, D]
            
            V_orth = V_agg - V_proj # [B, K, D]
        else:
            V_orth = V_agg # Fallback to direct addition
            
        self.last_V_orth = V_orth # Expose for testing
        
        # Update S with decay to prevent orthogonal space saturation
        S_next = (1 - self.decay_lambda) * S_prev + self.eta * V_orth
        
        S_time_next = S_time_prev
        if self.pe_mode == 'timestamps':
            # r_write max prob over time
            max_write_prob = r_write.max(dim=1)[0] # [B, K]
            # Soft update timestamp
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
            
        q_t = self.W_q(x_t) # [B, 1, D]
        v_t = self.W_v(x_t) # [B, 1, D]
        
        # 1. Read
        read_logits = torch.einsum('btd,bkd->btk', q_t, S_prev) / (self.dim ** 0.5)
        topk_logits, topk_idx = torch.topk(read_logits, self.kr, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)
        r_read = torch.zeros_like(read_logits).scatter(-1, topk_idx, topk_probs)
        context_t = torch.einsum('btk,bkd->btd', r_read, S_prev)
        
        # 2. Bypass
        if self.use_bypass:
            instr_prob = torch.sigmoid(self.W_m(q_t))
        else:
            instr_prob = torch.zeros(B, 1, 1, device=x_t.device)
            
        # Update KV cache
        if kv_cache is not None:
            K_cache, V_cache = kv_cache
            K_cache = torch.cat([K_cache, q_t], dim=1) # Using Q as K for simplicity as in original code
            V_cache = torch.cat([V_cache, v_t], dim=1)
        else:
            K_cache, V_cache = q_t, v_t
            
        # Cross attention to cache (all tokens in cache are valid for current token)
        bypass_out_t = F.scaled_dot_product_attention(q_t, K_cache, V_cache, is_causal=False)
        out_t = (1 - instr_prob) * context_t + instr_prob * bypass_out_t
        
        # 3. Write
        sim = F.cosine_similarity(v_t.unsqueeze(2), S_prev.unsqueeze(1), dim=-1)
        novelty = 1.0 - sim.max(dim=-1)[0]
        write_logits = self.W_n(torch.cat([novelty.unsqueeze(-1), q_t], dim=-1))
        r_write = torch.sigmoid(write_logits)
        
        # In step mode, V_agg is just weighted v_t
        V_agg = torch.einsum('btk,btd->bkd', r_write, v_t)
        
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
