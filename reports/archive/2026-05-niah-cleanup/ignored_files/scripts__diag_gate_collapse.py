"""
诊断门控崩溃：加载检查点，逐层分析 ST/MH 分支输出幅度、门控权重、slot 写入状态
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"设备: {DEVICE}")

# 加载最新检查点
ckpt_path = PROJECT_ROOT / "models" / "hybrid_lm" / "best_model.pt"
ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)

config = ckpt['config']
vocab_size = ckpt['vocab_size']
step = ckpt.get('step', 'N/A')
best_ppl = ckpt.get('best_ppl', 'N/A')
print(f"Step: {step}, Best PPL: {best_ppl}")
print(f"Config: dim={config['dim']}, n_layers={config['n_layers']}, n_heads={config['n_heads']}, slots={config['slots']}")
print()

# 导入模型
from scripts.pretrain_hybrid_lm import HybridLanguageModel

dim = config['dim']
n_layers = config['n_layers']
n_heads = config['n_heads']
slots = config['slots']
seq_len = config.get('seq_len', 512)

model = HybridLanguageModel(
    vocab_size=vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads,
    slots=slots, chunk_size=seq_len
).to(DEVICE)

# 加载权重
missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
print(f"加载检查点: missing={len(missing)}, unexpected={len(unexpected)}")
if missing:
    print(f"  Missing keys (前5): {missing[:5]}")
print()

sd = ckpt['model_state_dict']

# ============================================================================
# 1. 分析融合门控权重
# ============================================================================
print("=" * 80)
print("1. 融合门控权重分析")
print("=" * 80)
print()

for i in range(n_layers):
    w_key = f'fuse_gates.{i}.weight'
    b_key = f'fuse_gates.{i}.bias'
    if w_key in sd:
        w = sd[w_key]
        b = sd[b_key]
        # gate_logits = [h_st, h_mh] @ W + b
        # softmax 后 ST 权重 = exp(logit_st) / (exp(logit_st) + exp(logit_mh))
        # 当 bias[0] >> bias[1] 时，ST 占主导
        print(f"Layer {i}:")
        print(f"  weight shape: {w.shape}")
        print(f"  bias: ST={b[0].item():.4f}, MH={b[1].item():.4f}")
        print(f"  bias diff (ST-MH): {(b[0] - b[1]).item():.4f}")
        # 如果 bias diff > 3, softmax 后 ST > 0.95
        if b[0] - b[1] > 3:
            print(f"  ⚠️  bias diff > 3, ST 权重 > 0.95 (门控崩溃)")
        print(f"  weight norm: ST_col={w[:, 0].norm().item():.4f}, MH_col={w[:, 1].norm().item():.4f}")
        print(f"  weight row norm range: [{w.norm(dim=1).min().item():.4f}, {w.norm(dim=1).max().item():.4f}]")
        print()

# ============================================================================
# 2. 分析 token_write_gate 参数
# ============================================================================
print("=" * 80)
print("2. token_write_gate 参数分析")
print("=" * 80)
print()

for i in range(n_layers):
    w_key = f'mh_layers.{i}.token_write_gate.weight'
    b_key = f'mh_layers.{i}.token_write_gate.bias'
    if w_key in sd:
        w = sd[w_key]
        b = sd[b_key]
        sigmoid_b = torch.sigmoid(b).item()
        print(f"Layer {i}: bias={b.item():.4f}, sigmoid(bias)={sigmoid_b:.4f}, weight_std={w.std().item():.6f}")
print()

# ============================================================================
# 3. 分析 mh_out_norms 和 mh_scales
# ============================================================================
print("=" * 80)
print("3. mh_out_norms 和 mh_scales 参数分析")
print("=" * 80)
print()

for i in range(n_layers):
    scale_key = f'mh_scales.{i}'
    if scale_key in sd:
        scale = sd[scale_key]
        print(f"Layer {i}: mh_scale={scale.item():.6f}")
    else:
        print(f"Layer {i}: mh_scale 不存在（旧模型）")
print()

# ============================================================================
# 4. 前向传播分析：逐层对比 ST/MH 输出幅度
# ============================================================================
print("=" * 80)
print("4. 前向传播：逐层对比 ST/MH 输出幅度")
print("=" * 80)
print()

model.eval()
batch_size = 2
torch.manual_seed(42)
x_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)

with torch.no_grad():
    # 手动逐层前向传播
    positions = torch.arange(seq_len, device=DEVICE)
    h = model.tok_embedding(x_tokens) + model.pos_embedding(positions)
    
    for i in range(n_layers):
        # ST 分支
        causal_mask = model._get_causal_mask(seq_len, DEVICE)
        h_st = model.st_layers[i](h, src_mask=causal_mask, is_causal=True)
        h_st = model.st_projs[i](h_st)
        
        # MH 分支
        states_i = model.mh_layers[i].init_state(batch_size, device=DEVICE, dtype=torch.float32)
        h_mh, _, _ = model.mh_layers[i](h, state=states_i, return_aux=True)
        h_mh_normed = model.mh_out_norms[i](h_mh) * model.mh_scales[i]
        
        # 统计
        st_norm = h_st.norm(dim=-1).mean().item()
        mh_norm = h_mh.norm(dim=-1).mean().item()
        mh_normed_norm = h_mh_normed.norm(dim=-1).mean().item()
        st_std = h_st.std().item()
        mh_std = h_mh.std().item()
        mh_normed_std = h_mh_normed.std().item()
        
        # 融合门控
        gate_input = torch.cat([h_st, h_mh_normed], dim=-1)
        gate_logits = model.fuse_gates[i](gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)
        st_w = gate_weights[..., 0].mean().item()
        mh_w = gate_weights[..., 1].mean().item()
        
        print(f"Layer {i}:")
        print(f"  ST:  norm={st_norm:.4f}, std={st_std:.4f}")
        print(f"  MH:  norm={mh_norm:.4f}, std={mh_std:.4f} (原始)")
        print(f"  MH:  norm={mh_normed_norm:.4f}, std={mh_normed_std:.4f} (norm+scale后)")
        print(f"  MH/ST norm ratio: {mh_norm/st_norm:.4f} (原始), {mh_normed_norm/st_norm:.4f} (norm+scale后)")
        print(f"  Gate: ST={st_w:.4f}, MH={mh_w:.4f}")
        print()

# ============================================================================
# 5. 分析 slot 状态
# ============================================================================
print("=" * 80)
print("5. Slot 状态分析（完整前向传播后）")
print("=" * 80)
print()

with torch.no_grad():
    logits, final_states, gate_info = model(x_tokens, return_gate_info=True)
    
    for i in range(n_layers):
        s = final_states[i]
        usage = s.usage[0, 0]
        confidence = s.confidence[0, 0]
        age = s.age[0, 0]
        
        used_slots = (usage > 0.1).sum().item()
        total_slots = usage.shape[0]
        
        print(f"Layer {i}:")
        print(f"  usage: mean={usage.mean().item():.4f}, max={usage.max().item():.4f}, "
              f"used(>0.1)={used_slots}/{total_slots}")
        print(f"  confidence: mean={confidence.mean().item():.4f}, max={confidence.max().item():.4f}")
        print(f"  age: mean={age.mean().item():.2f}, max={age.max().item():.2f}")
        print(f"  slot_k norm: mean={s.slot_k[0,0].norm(dim=-1).mean().item():.4f}")
        print(f"  slot_v norm: mean={s.slot_v[0,0].norm(dim=-1).mean().item():.4f}")
        print()

# ============================================================================
# 6. 关键诊断：MHDSRA2 内部分析
# ============================================================================
print("=" * 80)
print("6. MHDSRA2 内部分析：slot_out vs local_out vs 融合输出")
print("=" * 80)
print()

with torch.no_grad():
    h = model.tok_embedding(x_tokens) + model.pos_embedding(positions)
    
    for i in range(n_layers):
        mh_layer = model.mh_layers[i]
        states_i = mh_layer.init_state(batch_size, device=DEVICE, dtype=torch.float32)
        
        # 手动调用 MHDSRA2 内部
        q, k, v = mh_layer.qkv(h).chunk(3, dim=-1)
        q_heads = mh_layer._to_heads(q)
        k_heads = mh_layer._to_heads(k)
        v_heads = mh_layer._to_heads(v)
        
        # Slot read
        slot_out, read_aux = mh_layer._slot_read(q_heads, states_i)
        slot_out_proj = mh_layer._to_dim(slot_out)
        
        # Local attention
        if mh_layer.cfg.use_local:
            local_out = mh_layer._local_attn(q_heads, k_heads, v_heads)
            local_out_proj = mh_layer._to_dim(local_out)
        else:
            local_out_proj = torch.zeros_like(slot_out_proj)
        
        # Fuse gate
        fuse_input = torch.cat([slot_out, local_out], dim=-1)
        fuse_logits = mh_layer.fuse_gate(fuse_input)
        fuse_weights = F.softmax(fuse_logits, dim=-1)
        
        slot_gate_w = fuse_weights[..., 0].mean().item()
        local_gate_w = fuse_weights[..., 1].mean().item()
        
        print(f"Layer {i}:")
        print(f"  slot_out: norm={slot_out_proj.norm(dim=-1).mean().item():.4f}, std={slot_out_proj.std().item():.6f}")
        print(f"  local_out: norm={local_out_proj.norm(dim=-1).mean().item():.4f}, std={local_out_proj.std().item():.6f}")
        print(f"  MH fuse gate: slot={slot_gate_w:.4f}, local={local_gate_w:.4f}")
        print(f"  slot_out/local_out norm ratio: {slot_out_proj.norm(dim=-1).mean().item() / (local_out_proj.norm(dim=-1).mean().item() + 1e-8):.4f}")
        print()

print("诊断完成!")
