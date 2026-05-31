"""
诊断 conflict 随层深递增的根本原因
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
print("诊断 conflict 随层深递增的原因")
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
        k_heads = mh._to_heads(k)
        
        # 计算 new_k（模拟写入前的聚合）
        slot_k = states_i.slot_k.to(dtype=k_heads.dtype)
        kn = F.normalize(k_heads, dim=-1)
        sk = F.normalize(slot_k, dim=-1)
        sim = torch.einsum("bhtd,bhkd->bhtk", kn, sk)
        max_sim = sim.max(dim=-1).values
        novelty = (1.0 - max_sim).clamp(0.0, 1.0)
        
        # 分析 max_sim 的分布
        print(f"Layer {i}:")
        print(f"  max_sim: mean={max_sim.mean().item():.4f}, min={max_sim.min().item():.4f}, max={max_sim.max().item():.4f}")
        print(f"  novelty: mean={novelty.mean().item():.4f}, min={novelty.min().item():.4f}, max={novelty.max().item():.4f}")
        
        # 检查 slot_k 的内容多样性
        slot_k_flat = slot_k[0, 0]  # [slots, d_head]
        slot_k_normed = F.normalize(slot_k_flat, dim=-1)
        
        # 计算 slot_k 之间的 pairwise cosine similarity
        # 如果多样性高，相似度应该接近 0
        slot_cos_sim = torch.mm(slot_k_normed, slot_k_normed.t())
        off_diag = slot_cos_sim[~torch.eye(len(slot_k_normed), dtype=bool, device=DEVICE)]
        print(f"  slot_k 多样性: mean_sim={off_diag.mean().item():.4f}, max_sim={off_diag.max().item():.4f}")
        
        # 计算 new_k 和 slot_k 之间的相似度
        # 对每个 token，计算与所有 slot 的最大相似度
        token_slot_sim = torch.einsum("bhtd,bhkd->bhtk", kn, sk)
        token_max_sim = token_slot_sim.max(dim=-1).values  # [B, H, T]
        print(f"  token-slot max_sim: mean={token_max_sim.mean().item():.4f}, min={token_max_sim.min().item():.4f}")
        
        # 逐 token 检查
        for pos in [0, seq_len//4, seq_len//2, seq_len-1]:
            pos_sim = token_slot_sim[0, :, pos]  # [H, slots]
            top_slot = pos_sim.max(dim=-1).values  # [H]
            print(f"    pos={pos}: token_max_sim={top_slot.mean().item():.4f}")
        
        # forward 并获取 conflict
        _, states_after, aux = mh(h, state=states_i, return_aux=True)
        ws = aux.get("write_stats", {})
        conflict = ws.get("conflict_mean", torch.tensor(0)).item()
        write_gate = ws.get("write_gate_mean", torch.tensor(0)).item()
        forget_gate = ws.get("forget_gate_mean", torch.tensor(0)).item()
        
        print(f"  conflict (实测): {conflict:.4f}")
        print(f"  write_gate: {write_gate:.4f}")
        print(f"  forget_gate: {forget_gate:.4f}")
        print()

# ============================================================================
# 关键测试：如果降低 conflict_protection_coef 或 forget_conflict，write_gate 会怎样？
# ============================================================================
print("=" * 80)
print("模拟不同 conflict 参数下的 write_gate")
print("=" * 80)
print()

# 假设 conflict=0.966, eta=0.25, max_update=0.5
conflict_val = 0.966
eta = 0.25
max_update = 0.5
write_mass = 0.5  # 假设的 write_mass

for conflict_protection_coef in [0.3, 0.1, 0.0]:
    reinforced_mass = write_mass  # 简化
    write_gate_base = 1 - torch.exp(torch.tensor(-eta * reinforced_mass)).item()
    write_gate = write_gate_base * max_update * (1 - conflict_protection_coef * conflict_val)
    write_gate = max(0, min(write_gate, max_update))
    
    for forget_conflict in [0.2, 0.05, 0.0]:
        forget_base = 0.001
        forget = forget_base + forget_conflict * write_gate * conflict_val
        
        print(f"  conflict_protection_coef={conflict_protection_coef}, forget_conflict={forget_conflict}:")
        print(f"    write_gate={write_gate:.4f}, forget={forget:.6f}")
