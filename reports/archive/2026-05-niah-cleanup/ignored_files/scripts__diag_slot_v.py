"""
快速诊断：为什么 slot_v 幅度远小于 v_heads
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

# 只分析第一层
mh = model.mh_layers[0]
states = mh.init_state(batch_size, device=DEVICE, dtype=torch.float32)

with torch.no_grad():
    h = model.tok_embedding(x_tokens) + model.pos_embedding(torch.arange(seq_len, device=DEVICE))
    
    q, k, v = mh.qkv(h).chunk(3, dim=-1)
    k_heads = mh._to_heads(k)
    v_heads = mh._to_heads(v)
    
    print(f"v_heads: std={v_heads.std().item():.6f}, mean_norm={v_heads.norm(dim=-1).mean().item():.4f}")
    print(f"slot_v (初始): std={states.slot_v.std().item():.6f}")
    print()
    
    # 逐步前向传播，观察 slot_v 的变化
    chunk_size = 64
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = x_tokens[:, start:end]
        
        h_chunk = model.tok_embedding(chunk) + model.pos_embedding(torch.arange(end-start, device=DEVICE))
        _, states, aux = mh(h_chunk, state=states, return_aux=True)
        
        ws = aux.get("write_stats", {})
        wg = ws.get("write_gate_mean", 0)
        fg = ws.get("forget_gate_mean", 0)
        wm = ws.get("write_mass_mean", 0)
        
        slot_v_std = states.slot_v.std().item()
        slot_v_norm = states.slot_v.norm(dim=-1).mean().item()
        
        print(f"  Chunk {start//chunk_size}: slot_v std={slot_v_std:.6f}, norm={slot_v_norm:.4f}, "
              f"write_gate={wg:.4f}, forget_gate={fg:.4f}, write_mass={wm:.4f}")
    
    print()
    print(f"最终 slot_v: std={states.slot_v.std().item():.6f}")
    print(f"v_heads: std={v_heads.std().item():.6f}")
    print(f"ratio: {states.slot_v.std().item() / v_heads.std().item():.4f}")
    print()
    
    # 关键：检查 write_gate 和 forget_gate 的关系
    # slot_v_next = (1 - forget) * slot_v + write_gate * new_v
    # 如果 forget ≈ write_gate，那么 slot_v_next ≈ (1-wg) * slot_v + wg * new_v
    # 稳态时 slot_v ≈ new_v
    # 但如果 forget > write_gate，slot_v 会衰减到 0
    
    # 检查 new_v 的幅度
    # new_v = agg_v / mass_safe
    # agg_v 是 scatter_add 后的结果，mass_safe 是写入的总权重
    # 如果多个 token 写入同一个 slot，agg_v 是它们的加权和，mass_safe 是权重和
    # new_v ≈ weighted_mean(v)，幅度应该接近 v_heads
    
    # 但 write_gate 限制了更新幅度
    # slot_v_next = (1 - forget) * slot_v + write_gate * new_v
    # 如果 max_update = 0.5，write_gate 最大 0.5
    # 稳态: slot_v = new_v * write_gate / (forget + write_gate)
    # 如果 forget ≈ write_gate ≈ 0.3，slot_v ≈ 0.5 * new_v
    
    print("理论分析:")
    print("  slot_v_next = (1 - forget) * slot_v + write_gate * new_v")
    print("  稳态: slot_v ≈ new_v * write_gate / (forget + write_gate)")
    print("  如果 forget ≈ write_gate ≈ 0.3, slot_v ≈ 0.5 * new_v")
    print("  如果 max_update = 0.5, write_gate 最大 0.5")
    print()
    
    # 检查 max_update 参数
    print(f"max_update: {mh.cfg.max_update}")
    print(f"eta: {mh.cfg.eta}")
    print(f"forget_base: {mh.cfg.forget_base}")
    print(f"forget_conflict: {mh.cfg.forget_conflict}")
