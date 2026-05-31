"""
精简诊断：直接检查 read_mass 和 slot read 质量
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
print("MHDSRA2 内部 read_mass 分析（正确路径）")
print("=" * 80)
print()

with torch.no_grad():
    positions = torch.arange(seq_len, device=DEVICE)
    h = model.tok_embedding(x_tokens) + model.pos_embedding(positions)
    
    for i in range(model.n_layers):
        mh = model.mh_layers[i]
        states_i = mh.init_state(batch_size, device=DEVICE, dtype=torch.float32)
        
        h_mh, new_states, aux = mh(h, state=states_i, return_aux=True)
        
        # 正确的 aux 结构
        print(f"Layer {i}: aux keys = {list(aux.keys())}")
        
        gates_mean = aux.get("gates_mean", None)
        read_mass = aux.get("read_mass", None)
        slot_usage = aux.get("slot_usage", None)
        write_stats = aux.get("write_stats", None)
        
        if gates_mean is not None:
            print(f"  gates_mean: {gates_mean.tolist()} (slot, local, retrieval)")
        
        if read_mass is not None:
            rm = read_mass[0, 0]
            nonzero = (rm > 1e-6).sum().item()
            total = rm.sum().item()
            top_slot = rm.argmax().item()
            top_mass = rm.max().item()
            print(f"  read_mass: nonzero={nonzero}/256, total={total:.4f}, top_slot={top_slot}, top_mass={top_mass:.6f}")
            if nonzero > 0:
                top5_vals, top5_idx = rm.topk(5)
                print(f"  read_mass top-5: {list(zip(top5_idx.tolist(), [f'{v:.4f}' for v in top5_vals.tolist()]))}")
        else:
            print(f"  read_mass: None")
        
        if slot_usage is not None:
            su = slot_usage[0, 0]
            print(f"  slot_usage: mean={su.mean().item():.4f}, max={su.max().item():.4f}, used(>0.1)={(su > 0.1).sum().item()}/256")
        
        if write_stats is not None:
            print(f"  write_stats: {write_stats}")
        
        print()

# ============================================================================
# 关键测试：直接调用 _slot_read 检查 logits
# ============================================================================
print("=" * 80)
print("直接调用 _slot_read 检查 logits 和 probs")
print("=" * 80)
print()

with torch.no_grad():
    h = model.tok_embedding(x_tokens) + model.pos_embedding(positions)
    
    for i in range(model.n_layers):
        mh = model.mh_layers[i]
        states_i = mh.init_state(batch_size, device=DEVICE, dtype=torch.float32)
        
        q, k, v = mh.qkv(h).chunk(3, dim=-1)
        q_heads = mh._to_heads(q)
        
        # 直接调用 _slot_read
        slot_out, read_aux = mh._slot_read(q_heads, states_i)
        
        # 检查 read_aux
        print(f"Layer {i}:")
        print(f"  read_aux keys: {list(read_aux.keys())}")
        
        read_idx = read_aux["read_idx"]
        read_probs = read_aux["read_probs"]
        read_mass = read_aux["read_mass"]
        read_logits_top = read_aux["read_logits_top"]
        
        print(f"  read_idx shape: {read_idx.shape}, sample: {read_idx[0, 0, 0, :4].tolist()}")
        print(f"  read_probs shape: {read_probs.shape}, sample: {read_probs[0, 0, 0, :4].tolist()}")
        print(f"  read_logits_top shape: {read_logits_top.shape}, sample: {read_logits_top[0, 0, 0, :4].tolist()}")
        
        rm = read_mass[0, 0]
        nonzero = (rm > 1e-6).sum().item()
        total = rm.sum().item()
        print(f"  read_mass: nonzero={nonzero}/256, total={total:.4f}")
        
        if nonzero > 0:
            top5_vals, top5_idx = rm.topk(5)
            print(f"  read_mass top-5: {list(zip(top5_idx.tolist(), [f'{v:.6f}' for v in top5_vals.tolist()]))}")
        else:
            print(f"  🔴 read_mass 全部为 0！")
            
            # 手动计算 logits
            slot_k = states_i.slot_k.to(dtype=q_heads.dtype)
            slot_v = states_i.slot_v.to(dtype=q_heads.dtype)
            
            # 不加 RoPE
            qn = F.normalize(q_heads, dim=-1)
            sk = F.normalize(slot_k, dim=-1)
            sim = torch.einsum("bhtd,bhkd->bhtk", qn, sk)
            tau = mh.log_tau_read.exp().to(dtype=q_heads.dtype)
            logits_no_rope = sim * tau
            
            print(f"  手动 logits (无RoPE): min={logits_no_rope.min().item():.4f}, max={logits_no_rope.max().item():.4f}, mean={logits_no_rope.mean().item():.4f}")
            print(f"  tau_read: {tau.item():.4f}")
            print(f"  conf_read_bias: {mh.cfg.conf_read_bias}")
            print(f"  age_read_penalty: {mh.cfg.age_read_penalty}")
            print(f"  confidence range: [{states_i.confidence.min().item():.4f}, {states_i.confidence.max().item():.4f}]")
            print(f"  age range: [{states_i.age.min().item():.2f}, {states_i.age.max().item():.2f}]")
            
            # 加上 confidence 和 age
            logits_with_conf = logits_no_rope + mh.cfg.conf_read_bias * states_i.confidence.to(dtype=q_heads.dtype).unsqueeze(2)
            logits_with_age = logits_with_conf - mh.cfg.age_read_penalty * torch.log1p(states_i.age).to(dtype=q_heads.dtype).unsqueeze(2)
            print(f"  手动 logits (含conf+age): min={logits_with_age.min().item():.4f}, max={logits_with_age.max().item():.4f}")
            
            # 检查 RoPE 的影响
            if mh.cfg.slot_pe == "rope" and mh.rotary is not None:
                B, H, T, D = q_heads.shape
                q_pos = torch.arange(states_i.position, states_i.position + T, device=DEVICE, dtype=torch.float32)
                q_pos = q_pos.view(1, 1, T, 1).expand(B, H, T, 1)
                k_pos = states_i.slot_positions.to(device=DEVICE, dtype=torch.float32).unsqueeze(-1)
                
                q_rope = mh.rotary.apply(q_heads, q_pos)
                sk_rope = mh.rotary.apply(slot_k, k_pos)
                
                qn_r = F.normalize(q_rope, dim=-1)
                sk_r = F.normalize(sk_rope, dim=-1)
                sim_rope = torch.einsum("bhtd,bhkd->bhtk", qn_r, sk_r)
                logits_rope = sim_rope * tau
                
                print(f"  手动 logits (含RoPE): min={logits_rope.min().item():.4f}, max={logits_rope.max().item():.4f}")
                print(f"  RoPE 后相似度: min={sim_rope.min().item():.6f}, max={sim_rope.max().item():.6f}, mean={sim_rope.mean().item():.6f}")
                print(f"  无RoPE相似度: min={sim.min().item():.6f}, max={sim.max().item():.6f}, mean={sim.mean().item():.6f}")
                
                # 检查 slot_positions
                print(f"  slot_positions: min={states_i.slot_positions.min().item():.1f}, max={states_i.slot_positions.max().item():.1f}")
                print(f"  state.position: {states_i.position}")
        
        print()
