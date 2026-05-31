"""
精确诊断：为什么 MHDSRA2 输出幅度远小于 ST
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ckpt_path = PROJECT_ROOT / "models" / "hybrid_lm" / "best_model.pt"
ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
config = ckpt['config']
vocab_size = ckpt['vocab_size']

from scripts.pretrain_hybrid_lm import HybridLanguageModel

model = HybridLanguageModel(
    vocab_size=vocab_size, dim=config['dim'], n_layers=config['n_layers'],
    n_heads=config['n_heads'], slots=config['slots'],
    chunk_size=config.get('seq_len', 512)
).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

batch_size = 2
seq_len = config.get('seq_len', 512)
torch.manual_seed(42)
x_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)

print("=" * 80)
print("精确诊断：MHDSRA2 输出幅度分析")
print("=" * 80)
print()

with torch.no_grad():
    positions = torch.arange(seq_len, device=DEVICE)
    h = model.tok_embedding(x_tokens) + model.pos_embedding(positions)
    
    for i in range(model.n_layers):
        mh = model.mh_layers[i]
        states_i = mh.init_state(batch_size, device=DEVICE, dtype=torch.float32)
        
        # QKV 投影
        q, k, v = mh.qkv(h).chunk(3, dim=-1)
        q_heads = mh._to_heads(q)
        k_heads = mh._to_heads(k)
        v_heads = mh._to_heads(v)
        
        # Slot read
        slot_out, read_aux = mh._slot_read(q_heads, states_i)
        slot_out_full = mh._from_heads(slot_out)
        slot_out_proj = mh.out_proj(slot_out_full)
        
        # Local attention
        local_out, _, _ = mh._local_attention(q_heads, k_heads, v_heads, states_i)
        local_out_full = mh._from_heads(local_out)
        local_out_proj = mh.out_proj(local_out_full)
        
        # MHDSRA2 内部融合 (fuse_gate 输入是 q, 不是 cat([slot_out, local_out]))
        fuse_logits = mh.fuse_gate(q_heads)
        position_norm = torch.arange(seq_len, device=DEVICE, dtype=q_heads.dtype) / max(seq_len, 1)
        position_feature = position_norm.view(1, seq_len, 1, 1).expand(1, -1, mh.heads, 3)
        position_feature = position_feature.transpose(1, 2)
        fuse_logits = fuse_logits + position_feature * 0.05
        gates = torch.sigmoid(fuse_logits)
        gate_mask = torch.ones_like(gates)
        if not mh.cfg.use_local or mh.cfg.local_window <= 0:
            gate_mask[..., 1] = 0.0
        gate_mask[..., 2] = 0.0  # no retrieval
        gates = gates * gate_mask
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-8)
        
        slot_w = gates[..., 0:1]
        local_w = gates[..., 1:2]
        
        # 融合后的 head-space 输出
        fused_heads = slot_w * slot_out + local_w * local_out
        fused_full = mh._from_heads(fused_heads)
        fused_proj = mh.out_proj(fused_full)
        
        # ST 分支
        causal_mask = model._get_causal_mask(seq_len, DEVICE)
        h_st = model.st_layers[i](h, src_mask=causal_mask, is_causal=True)
        h_st = model.st_projs[i](h_st)
        
        print(f"Layer {i}:")
        print(f"  === MHDSRA2 内部 ===")
        print(f"  slot_out (head-space): std={slot_out.std().item():.6f}")
        print(f"  local_out (head-space): std={local_out.std().item():.6f}")
        print(f"  fuse_gate: slot_w={slot_w.mean().item():.4f}, local_w={local_w.mean().item():.4f}")
        print(f"  fused_heads: std={fused_heads.std().item():.6f}")
        print(f"  === 经过 out_proj 后 ===")
        print(f"  slot_out_proj: std={slot_out_proj.std().item():.6f}")
        print(f"  local_out_proj: std={local_out_proj.std().item():.6f}")
        print(f"  fused_proj (MHDSRA2 输出): std={fused_proj.std().item():.6f}")
        print(f"  === ST 分支 ===")
        print(f"  h_st: std={h_st.std().item():.6f}")
        print(f"  === 幅度比 ===")
        print(f"  MH/ST std ratio: {fused_proj.std().item() / h_st.std().item():.4f}")
        print(f"  slot_out_proj/fused_proj: {slot_out_proj.std().item() / fused_proj.std().item():.4f}")
        print(f"  local_out_proj/fused_proj: {local_out_proj.std().item() / fused_proj.std().item():.4f}")
        
        # 检查 out_proj 权重
        out_w = mh.out_proj.weight
        print(f"  === out_proj 权重 ===")
        print(f"  out_proj weight norm: {out_w.norm().item():.4f}")
        print(f"  out_proj weight std: {out_w.std().item():.6f}")
        
        # 检查 st_proj 权重
        st_w = model.st_projs[i].weight
        print(f"  st_proj weight norm: {st_w.norm().item():.4f}")
        print(f"  st_proj weight std: {st_w.std().item():.6f}")
        print(f"  out_proj/st_proj weight norm ratio: {out_w.norm().item() / st_w.norm().item():.4f}")
        
        # 关键：检查 slot_out 和 local_out 是否互相抵消
        # 如果 slot_out 和 local_out 方向相反，融合后会互相抵消
        slot_flat = slot_out_proj.reshape(-1)
        local_flat = local_out_proj.reshape(-1)
        cos_sim = F.cosine_similarity(slot_flat.unsqueeze(0), local_flat.unsqueeze(0)).item()
        print(f"  === 信号抵消检查 ===")
        print(f"  slot_out_proj vs local_out_proj cosine similarity: {cos_sim:.4f}")
        if cos_sim < -0.3:
            print(f"  🔴 slot 和 local 输出方向相反！融合后互相抵消")
        elif cos_sim < 0.1:
            print(f"  🟡 slot 和 local 输出几乎正交，没有明显抵消")
        else:
            print(f"  ✅ slot 和 local 输出方向一致")
        print()
