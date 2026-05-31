"""
深度诊断：MHDSRA2 内部 slot_out vs local_out 质量
简化版：直接使用 forward + return_aux 获取内部信息
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
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
print("1. MHDSRA2 内部：slot read 质量分析")
print("=" * 80)
print()

with torch.no_grad():
    positions = torch.arange(seq_len, device=DEVICE)
    h = model.tok_embedding(x_tokens) + model.pos_embedding(positions)
    
    for i in range(model.n_layers):
        mh = model.mh_layers[i]
        states_i = mh.init_state(batch_size, device=DEVICE, dtype=torch.float32)
        
        # 完整 MHDSRA2 forward
        h_mh, new_states, aux = mh(h, state=states_i, return_aux=True)
        
        # 从 aux 获取内部信息
        read_aux = aux.get("read_aux", {})
        write_stats = aux.get("write_stats", {})
        
        # slot_out 和 local_out 的统计
        # 从 forward 代码中，MHDSRA2 的输出是 fuse_gate 融合后的结果
        # 我们需要分别看 slot 和 local 的贡献
        
        # 检查 read_mass 分布
        if "read_mass" in read_aux:
            rm = read_aux["read_mass"][0, 0]
            top_slot = rm.argmax().item()
            top_mass = rm.max().item()
            total_mass = rm.sum().item()
            nonzero = (rm > 1e-6).sum().item()
            entropy = -(rm[rm > 1e-6] * torch.log(rm[rm > 1e-6] + 1e-8)).sum().item()
            max_entropy = torch.log(torch.tensor(float(nonzero))).item() if nonzero > 1 else 0
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            top_slot = -1; top_mass = 0; total_mass = 0; nonzero = 0; norm_entropy = 0
        
        # 检查 write_stats
        token_gate_mean = write_stats.get("token_gate_mean", 0)
        write_mass_mean = write_stats.get("write_mass_mean", 0)
        
        # MHDSRA2 输出统计
        mh_std = h_mh.std().item()
        mh_norm = h_mh.norm(dim=-1).mean().item()
        
        # 对比 ST 输出
        causal_mask = model._get_causal_mask(seq_len, DEVICE)
        h_st = model.st_layers[i](h, src_mask=causal_mask, is_causal=True)
        h_st = model.st_projs[i](h_st)
        st_std = h_st.std().item()
        st_norm = h_st.norm(dim=-1).mean().item()
        
        print(f"Layer {i}:")
        print(f"  MH out: std={mh_std:.6f}, norm={mh_norm:.4f}")
        print(f"  ST out: std={st_std:.6f}, norm={st_norm:.4f}")
        print(f"  MH/ST std ratio: {mh_std/st_std:.4f}")
        print(f"  read_mass: top_slot={top_slot}, top_mass={top_mass:.4f}, "
              f"total={total_mass:.4f}, nonzero={nonzero}/256")
        print(f"  read_mass normalized_entropy: {norm_entropy:.4f} (1.0=均匀分布)")
        print(f"  token_gate_mean: {token_gate_mean:.4f}")
        print(f"  write_mass_mean: {write_mass_mean:.6f}")
        print()

# ============================================================================
# 2. 关键测试：MHDSRA2 输出 vs 纯 local attention 输出
# ============================================================================
print("=" * 80)
print("2. MHDSRA2 输出 vs 纯 local attention 输出")
print("=" * 80)
print()

with torch.no_grad():
    h = model.tok_embedding(x_tokens) + model.pos_embedding(positions)
    
    for i in range(model.n_layers):
        mh = model.mh_layers[i]
        states_i = mh.init_state(batch_size, device=DEVICE, dtype=torch.float32)
        
        # 正常 MHDSRA2
        h_mh_normal, _, _ = mh(h, state=states_i, return_aux=True)
        
        # 临时将 fuse_gate 强制为 local-only
        q, k, v = mh.qkv(h).chunk(3, dim=-1)
        q_heads = mh._to_heads(q)
        k_heads = mh._to_heads(k)
        v_heads = mh._to_heads(v)
        
        # 只用 local attention
        local_out, _, _ = mh._local_attention(q_heads, k_heads, v_heads, states_i)
        local_out_full = mh._from_heads(local_out)
        h_mh_local_only = mh.out_proj(local_out_full)
        
        # 只用 slot attention
        slot_out, read_aux = mh._slot_read(q_heads, states_i)
        slot_out_full = mh._from_heads(slot_out)
        h_mh_slot_only = mh.out_proj(slot_out_full)
        
        diff_local = (h_mh_normal - h_mh_local_only).norm().item()
        diff_slot = (h_mh_normal - h_mh_slot_only).norm().item()
        normal_norm = h_mh_normal.norm().item()
        
        print(f"Layer {i}:")
        print(f"  normal MH:  std={h_mh_normal.std().item():.6f}, norm={normal_norm:.4f}")
        print(f"  local-only: std={h_mh_local_only.std().item():.6f}, norm={h_mh_local_only.norm().item():.4f}")
        print(f"  slot-only:  std={h_mh_slot_only.std().item():.6f}, norm={h_mh_slot_only.norm().item():.4f}")
        print(f"  diff(normal, local-only): {diff_local:.6f} ({diff_local/normal_norm*100:.2f}%)")
        print(f"  diff(normal, slot-only):  {diff_slot:.6f} ({diff_slot/normal_norm*100:.2f}%)")
        
        slot_contribution = 1.0 - diff_local / normal_norm if normal_norm > 0 else 0
        local_contribution = 1.0 - diff_slot / normal_norm if normal_norm > 0 else 0
        print(f"  slot 贡献度: ~{max(0, slot_contribution)*100:.1f}%")
        print(f"  local 贡献度: ~{max(0, local_contribution)*100:.1f}%")
        print()

# ============================================================================
# 3. 检查 slot_k 和 slot_v 的信息量
# ============================================================================
print("=" * 80)
print("3. Slot 内容分析：slot_k 和 slot_v 的信息量")
print("=" * 80)
print()

with torch.no_grad():
    h = model.tok_embedding(x_tokens) + model.pos_embedding(positions)
    
    for i in range(model.n_layers):
        mh = model.mh_layers[i]
        states_i = mh.init_state(batch_size, device=DEVICE, dtype=torch.float32)
        _, new_states, _ = mh(h, state=states_i, return_aux=True)
        
        slot_k = new_states.slot_k[0, 0]
        slot_v = new_states.slot_v[0, 0]
        usage = new_states.usage[0, 0]
        
        # 找到被使用的 slot
        used_mask = usage > 0.1
        used_indices = used_mask.nonzero(as_tuple=True)[0].tolist()
        
        print(f"Layer {i}: used_slots={len(used_indices)}/{usage.shape[0]}")
        
        if used_indices:
            # 分析被使用 slot 的 key 和 value
            for idx in used_indices[:4]:
                k_norm = slot_k[idx].norm().item()
                v_norm = slot_v[idx].norm().item()
                u = usage[idx].item()
                print(f"  Slot {idx}: usage={u:.2f}, key_norm={k_norm:.6f}, value_norm={v_norm:.6f}")
            
            # 检查被使用 slot 的 key 之间的相似度
            used_k = slot_k[used_mask]
            if len(used_k) > 1:
                used_k_norm = F.normalize(used_k, dim=-1)
                sim_matrix = torch.mm(used_k_norm, used_k_norm.t())
                off_diag = sim_matrix[~torch.eye(len(used_k), dtype=bool, device=DEVICE)]
                print(f"  被使用 slot 的 key 间相似度: mean={off_diag.mean().item():.4f}, max={off_diag.max().item():.4f}")
        print()

print("诊断完成!")
