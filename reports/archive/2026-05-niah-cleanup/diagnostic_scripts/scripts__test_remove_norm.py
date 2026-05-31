
import sys
sys.path.insert(0, r'e:\Project\python\DSRA')
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dsra.mhdsra2.improved_dsra_mha import MultiHeadDSRA2, MHDSRA2Config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

ckpt = torch.load(r'e:\Project\python\DSRA\models\hybrid_lm\best_model.pt', map_location=device, weights_only=False)
config = ckpt['config']
print(f'配置: {config}')

vocab_size = ckpt['vocab_size']
dim = config['dim']
n_heads = config['n_heads']
slots = config['slots']

# 重建 MHDSRA2
mh_cfg = MHDSRA2Config(
    dim=dim, heads=n_heads, local_window=512, slot_pe='rope', slots=slots,
    use_retrieval=False
)
mh_layer = MultiHeadDSRA2(mh_cfg).to(device)

# 加载权重
mh_sd = {k[len('mh_layers.0.'):]: v for k, v in ckpt['model_state_dict'].items() if k.startswith('mh_layers.0.')}
mh_layer.load_state_dict(mh_sd, strict=False)
mh_layer.eval()

# 测试数据
torch.manual_seed(42)
batch_size = 2
seq_len = 128
x = torch.randn(batch_size, seq_len, dim, device=device)
states = mh_layer.init_state(batch_size, device=device, dtype=torch.float32)

print()
print('='*80)
print('测试1: 有归一化（原始）')
print('='*80)

with torch.no_grad():
    qkv = mh_layer.qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)
    q = mh_layer._to_heads(q)
    k = mh_layer._to_heads(k)
    v = mh_layer._to_heads(v)
    
    slot_k = states.slot_k.to(dtype=q.dtype)
    slot_v = states.slot_v.to(dtype=q.dtype)
    
    # 原始归一化
    qn_normed = F.normalize(q, dim=-1)
    sk_normed = F.normalize(slot_k, dim=-1)
    tau = mh_layer.log_tau_read.exp()
    logits_normed = torch.einsum("bhtd,bhkd->bhtk", qn_normed, sk_normed) * tau
    logits_normed = logits_normed + mh_cfg.conf_read_bias * states.confidence.unsqueeze(2)
    logits_normed = logits_normed - mh_cfg.age_read_penalty * torch.log1p(states.age).unsqueeze(2)
    
    r = min(mh_cfg.read_topk, mh_cfg.slots)
    top_logits_normed, top_idx = torch.topk(logits_normed, r, dim=-1)
    probs_normed = F.softmax(top_logits_normed, dim=-1)
    selected_v_normed = mh_layer._gather_slots(slot_v, top_idx)
    slot_out_normed = (probs_normed.unsqueeze(-1) * selected_v_normed).sum(dim=3)
    
    print(f'q std before norm: {q.std():.6f}')
    print(f'qn std after norm: {qn_normed.std():.6f}')
    print(f'sk std after norm: {sk_normed.std():.6f}')
    print(f'logits (with norm) - mean: {logits_normed.mean():.6f}, std: {logits_normed.std():.6f}')
    print(f'logits range: [{logits_normed.min():.4f}, {logits_normed.max():.4f}]')
    print(f'slot_out (with norm) - mean: {slot_out_normed.mean():.6f}, std: {slot_out_normed.std():.6f}')
    print(f'top probs entropy: {(-probs_normed * probs_normed.log()).sum(dim=-1).mean():.4f}')

print()
print('='*80)
print('测试2: 去除归一化')
print('='*80)

with torch.no_grad():
    # 去除归一化
    qn_unormed = q
    sk_unormed = slot_k
    logits_unormed = torch.einsum("bhtd,bhkd->bhtk", qn_unormed, sk_unormed) * tau
    logits_unormed = logits_unormed + mh_cfg.conf_read_bias * states.confidence.unsqueeze(2)
    logits_unormed = logits_unormed - mh_cfg.age_read_penalty * torch.log1p(states.age).unsqueeze(2)
    
    top_logits_unormed, top_idx_unormed = torch.topk(logits_unormed, r, dim=-1)
    probs_unormed = F.softmax(top_logits_unormed, dim=-1)
    selected_v_unormed = mh_layer._gather_slots(slot_v, top_idx_unormed)
    slot_out_unormed = (probs_unormed.unsqueeze(-1) * selected_v_unormed).sum(dim=3)
    
    print(f'qn std (no norm): {qn_unormed.std():.6f}')
    print(f'sk std (no norm): {sk_unormed.std():.6f}')
    print(f'logits (no norm) - mean: {logits_unormed.mean():.6f}, std: {logits_unormed.std():.6f}')
    print(f'logits range: [{logits_unormed.min():.4f}, {logits_unormed.max():.4f}]')
    print(f'slot_out (no norm) - mean: {slot_out_unormed.mean():.6f}, std: {slot_out_unormed.std():.6f}')
    print(f'top probs entropy: {(-probs_unormed * probs_unormed.log()).sum(dim=-1).mean():.4f}')
    
    print()
    print('⚠️ 关键发现')
    print(f'   - 无归一化时 logits std = {logits_unormed.std():.2f}，是有归一化的 {logits_unormed.std()/logits_normed.std():.1f}x')
    print(f'   - logits 范围从 [{logits_normed.min():.1f}, {logits_normed.max():.1f}] 扩大到 [{logits_unormed.min():.1f}, {logits_unormed.max():.1f}]')
    print(f'   - softmax 会变得极端（熵从 {(-probs_normed * probs_normed.log()).sum(dim=-1).mean():.3f} 降到 {(-probs_unormed * probs_unormed.log()).sum(dim=-1).mean():.3f}）')
    print()

print()
print('='*80)
print('测试3: 折中方案 - 保留归一化但添加增益')
print('='*80)

with torch.no_grad():
    # 归一化后乘以增益
    gain = 2.5
    qn_scaled = F.normalize(q, dim=-1) * gain
    sk_scaled = F.normalize(slot_k, dim=-1) * gain
    
    logits_scaled = torch.einsum("bhtd,bhkd->bhtk", qn_scaled, sk_scaled) * tau
    logits_scaled = logits_scaled + mh_cfg.conf_read_bias * states.confidence.unsqueeze(2)
    logits_scaled = logits_scaled - mh_cfg.age_read_penalty * torch.log1p(states.age).unsqueeze(2)
    
    top_logits_scaled, top_idx_scaled = torch.topk(logits_scaled, r, dim=-1)
    probs_scaled = F.softmax(top_logits_scaled, dim=-1)
    selected_v_scaled = mh_layer._gather_slots(slot_v, top_idx_scaled)
    slot_out_scaled = (probs_scaled.unsqueeze(-1) * selected_v_scaled).sum(dim=3)
    
    print(f'q scaled std: {qn_scaled.std():.6f}')
    print(f'sk scaled std: {sk_scaled.std():.6f}')
    print(f'logits scaled - mean: {logits_scaled.mean():.6f}, std: {logits_scaled.std():.6f}')
    print(f'slot_out scaled - mean: {slot_out_scaled.mean():.6f}, std: {slot_out_scaled.std():.6f}')
    print(f'top probs entropy: {(-probs_scaled * probs_scaled.log()).sum(dim=-1).mean():.4f}')

print()
print('='*80)
print('测试4: 完整模型对比')
print('='*80)

with torch.no_grad():
    # 原始完整模型
    result_normed = mh_layer(x, state=states, return_aux=True)
    if len(result_normed) == 3:
        y_normed, states_out_normed, aux_normed = result_normed
    else:
        y_normed, states_out_normed = result_normed
    
    print(f'原始完整模型 - mean: {y_normed.mean():.6f}, std: {y_normed.std():.6f}')
    
    # 现在临时替换 slot_read 为无归一化版本
    original_slot_read = mh_layer._slot_read
    
    def modified_slot_read(q, state):
        cfg = mh_layer.cfg
        slot_k = state.slot_k.to(dtype=q.dtype)
        slot_v = state.slot_v.to(dtype=q.dtype)
        
        if cfg.slot_pe == "rope" and mh_layer.rotary is not None and state.slot_positions is not None:
            B, H, T, D = q.shape
            q_pos = torch.arange(state.position, state.position + T, device=q.device, dtype=torch.float32)
            q_pos = q_pos.view(1, 1, T, 1).expand(B, H, T, 1)
            k_pos = state.slot_positions.to(device=q.device, dtype=torch.float32).unsqueeze(-1)
            q_rope = mh_layer.rotary.apply(q, q_pos)
            slot_k_rope = mh_layer.rotary.apply(slot_k, k_pos)
            # 去除归一化，但保留增益
            qn = q_rope
            sk = slot_k_rope
        else:
            qn = q
            sk = slot_k
        
        tau = mh_layer.log_tau_read.exp().to(dtype=q.dtype)
        logits = torch.einsum("bhtd,bhkd->bhtk", qn, sk) * tau
        logits = logits + cfg.conf_read_bias * state.confidence.to(dtype=q.dtype).unsqueeze(2)
        logits = logits - cfg.age_read_penalty * torch.log1p(state.age).to(dtype=q.dtype).unsqueeze(2)
        
        r = min(cfg.read_topk, cfg.slots)
        top_logits, top_idx = torch.topk(logits, r, dim=-1)
        probs = F.softmax(top_logits, dim=-1)
        selected_v = mh_layer._gather_slots(slot_v, top_idx)
        out = (probs.unsqueeze(-1) * selected_v).sum(dim=3)
        read_mass = mh_layer._scatter_mass(top_idx, probs, cfg.slots)
        
        return out, {"read_idx": top_idx, "read_probs": probs, "read_mass": read_mass}
    
    mh_layer._slot_read = modified_slot_read
    
    # 运行修改后的模型
    states = mh_layer.init_state(batch_size, device=device, dtype=torch.float32)
    result_unormed = mh_layer(x, state=states, return_aux=True)
    if len(result_unormed) == 3:
        y_unormed, states_out_unormed, aux_unormed = result_unormed
    else:
        y_unormed, states_out_unormed = result_unormed
    
    print(f'去除归一化后 - mean: {y_unormed.mean():.6f}, std: {y_unormed.std():.6f}')
    print(f'std提升倍数: {y_unormed.std()/y_normed.std():.2f}x')
    
    # 恢复
    mh_layer._slot_read = original_slot_read

print()
print('='*80)
print('总结')
print('='*80)
print()
print('1. 归一化的设计意图')
print('   - 将 q 和 k 投影到单位球上，使得点积等于余弦相似度')
print('   - 配合温度 tau，精确控制 slot attention 的锐度')
print('   - 让 slot similarity 具有可比性，不受向量尺度影响')
print()
print('2. 去除归一化的后果')
print('   - logits 的 std 会大幅增加（可能导致 softmax 极端化）')
print('   - 注意力会变得不稳定（某个大向量主导）')
print('   - slot_v 的输出尺度会大幅波动')
print()
print('3. 折中方案')
print('   - 保留归一化，但在 slot_out 后添加增益缩放')
print('   - 或者在 out_proj 后添加 LayerNorm 和可学习的缩放参数')

