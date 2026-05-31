"""
最终诊断：slot read 的 probs 分布是否过于均匀导致输出方差小
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
print("slot read probs 分布分析")
print("=" * 80)
print()

with torch.no_grad():
    positions = torch.arange(seq_len, device=DEVICE)
    h = model.tok_embedding(x_tokens) + model.pos_embedding(positions)
    
    for i in range(model.n_layers):
        mh = model.mh_layers[i]
        states_i = mh.init_state(batch_size, device=DEVICE, dtype=torch.float32)
        
        q, k, v = mh.qkv(h).chunk(3, dim=-1)
        q_heads = mh._to_heads(q)
        
        slot_out, read_aux = mh._slot_read(q_heads, states_i)
        
        read_probs = read_aux["read_probs"]  # [B, H, T, read_topk]
        read_idx = read_aux["read_idx"]
        
        # 分析 probs 分布
        probs = read_probs[0, 0]  # [T, read_topk]
        
        # 每个 token 的 top-1 prob
        top1_prob = probs[:, 0]
        # 每个 token 的 probs 熵
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(probs.shape[-1])))
        norm_entropy = entropy / max_entropy
        
        # 如果 top-1 prob 接近 1/read_topk = 0.125，说明分布过于均匀
        # 如果 norm_entropy 接近 1.0，说明分布过于均匀
        
        print(f"Layer {i}:")
        print(f"  read_topk: {probs.shape[-1]}")
        print(f"  top-1 prob: mean={top1_prob.mean().item():.4f}, min={top1_prob.min().item():.4f}, max={top1_prob.max().item():.4f}")
        print(f"  1/read_topk = {1.0/probs.shape[-1]:.4f} (均匀分布时的 top-1 prob)")
        print(f"  normalized entropy: mean={norm_entropy.mean().item():.4f} (1.0=完全均匀)")
        
        # 检查 slot_v 的方差
        slot_v = states_i.slot_v[0, 0]  # [slots, d_head]
        print(f"  slot_v: std={slot_v.std().item():.6f}, norm_mean={slot_v.norm(dim=-1).mean().item():.4f}")
        
        # 检查 local attention 的 v 的方差
        k_heads = mh._to_heads(k)
        v_heads = mh._to_heads(v)
        print(f"  v_heads: std={v_heads.std().item():.6f}")
        
        # 关键对比：slot_v vs v_heads 的方差
        print(f"  slot_v/v_heads std ratio: {slot_v.std().item() / v_heads.std().item():.4f}")
        
        # 检查 slot_out 的组成
        # slot_out = sum(probs * selected_v)
        # 如果 probs 均匀，slot_out ≈ mean(selected_v)，方差会很小
        # 如果 probs 集中，slot_out ≈ top-1 v，方差接近 v 的方差
        
        # 模拟：如果 probs 是 one-hot（只选 top-1），slot_out 会怎样？
        selected_v = mh._gather_slots(states_i.slot_v.to(dtype=q_heads.dtype), read_idx)
        top1_v = selected_v[0, 0, :, 0, :]  # [T, d_head] - 只取 top-1
        weighted_v = (probs.unsqueeze(-1) * selected_v[0, 0]).sum(dim=1)  # [T, d_head] - 加权平均
        
        print(f"  top1_v (one-hot read): std={top1_v.std().item():.6f}")
        print(f"  weighted_v (soft read): std={weighted_v.std().item():.6f}")
        print(f"  top1_v/weighted_v std ratio: {top1_v.std().item() / (weighted_v.std().item() + 1e-8):.2f}x")
        
        if norm_entropy.mean().item() > 0.9:
            print(f"  🔴 probs 分布过于均匀！导致 slot_out 方差过小")
        elif norm_entropy.mean().item() > 0.7:
            print(f"  🟡 probs 分布偏均匀，slot_out 方差偏小")
        else:
            print(f"  ✅ probs 分布有区分度")
        print()

# ============================================================================
# 关键测试：如果 read_topk=1，slot_out 的幅度会怎样？
# ============================================================================
print("=" * 80)
print("模拟 read_topk=1 时 slot_out 的幅度")
print("=" * 80)
print()

# 临时修改 read_topk
for i in range(model.n_layers):
    mh = model.mh_layers[i]
    original_topk = mh.cfg.read_topk
    mh.cfg.read_topk = 1
    
    with torch.no_grad():
        h = model.tok_embedding(x_tokens) + model.pos_embedding(positions)
        q, k, v = mh.qkv(h).chunk(3, dim=-1)
        q_heads = mh._to_heads(q)
        states_i = mh.init_state(batch_size, device=DEVICE, dtype=torch.float32)
        
        slot_out, read_aux = mh._slot_read(q_heads, states_i)
        slot_out_full = mh._from_heads(slot_out)
        slot_out_proj = mh.out_proj(slot_out_full)
        
        read_probs = read_aux["read_probs"][0, 0]
        top1_prob = read_probs[:, 0].mean().item()
        
        print(f"Layer {i} (read_topk=1): slot_out_proj std={slot_out_proj.std().item():.6f}, top1_prob={top1_prob:.4f}")
    
    mh.cfg.read_topk = original_topk

print()
print("诊断完成!")
