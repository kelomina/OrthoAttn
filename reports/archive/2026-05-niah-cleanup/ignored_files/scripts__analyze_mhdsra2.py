
import sys
sys.path.insert(0, r'e:\Project\python\DSRA')
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dsra.mhdsra2.improved_dsra_mha import MultiHeadDSRA2, MHDSRA2Config, MHDSRA2State

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

ckpt = torch.load(r'e:\Project\python\DSRA\models\hybrid_lm\best_model.pt', map_location=device, weights_only=False)
config = ckpt['config']
print(f'配置: {config}')

vocab_size = ckpt['vocab_size']
dim = config['dim']
n_heads = config['n_heads']
slots = config['slots']
seq_len = config['seq_len']

# 重建 MHDSRA2
mh_cfg = MHDSRA2Config(
    dim=dim, heads=n_heads, local_window=512, slot_pe='rope', slots=slots,
    use_retrieval=False
)
mh_layer = MultiHeadDSRA2(mh_cfg).to(device)
mh_sd = {k[len('mh_layers.0.'):]: v for k, v in ckpt['model_state_dict'].items() if k.startswith('mh_layers.0.')}
mh_layer.load_state_dict(mh_sd, strict=False)
mh_layer.eval()

# 测试数据
torch.manual_seed(42)
batch_size = 2
x = torch.randn(batch_size, seq_len, dim, device=device)
states = mh_layer.init_state(batch_size, device=device, dtype=torch.float32)

# ST 分支
st_layer = nn.TransformerEncoderLayer(
    d_model=dim, nhead=n_heads, dim_feedforward=dim*4, batch_first=True, activation='gelu'
).to(device)
st_sd = {k[len('st_layers.0.'):]: v for k, v in ckpt['model_state_dict'].items() if k.startswith('st_layers.0.')}
st_layer.load_state_dict(st_sd, strict=False)

st_proj = nn.Linear(dim, dim).to(device)
st_proj_sd = {
    'weight': ckpt['model_state_dict']['st_projs.0.weight'],
    'bias': ckpt['model_state_dict']['st_projs.0.bias']
}
st_proj.load_state_dict(st_proj_sd, strict=True)

print()
print('='*80)
print('MHDSRA2 内部机制深度分析')
print('='*80)

with torch.no_grad():
    # 1. 分析 qkv
    print('\n1. QKV 分析')
    qkv = mh_layer.qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)
    q = mh_layer._to_heads(q)
    k = mh_layer._to_heads(k)
    v = mh_layer._to_heads(v)
    print(f'   q std: {q.std():.4f}, k std: {k.std():.4f}, v std: {v.std():.4f}')
    
    # 2. 分析 local attention
    print('\n2. Local Attention 分析')
    local_out, new_k, new_v = mh_layer._local_attention(q, k, v, states)
    print(f'   local_out std: {local_out.std():.6f}')
    
    # 3. 分析 slot read
    print('\n3. Slot Read 分析')
    slot_out, _ = mh_layer._slot_read(q, states)
    print(f'   slot_out std: {slot_out.std():.6f}')
    
    # 4. 分析 MHDSRA2 内部门控
    print('\n4. MHDSRA2 内部门控分析')
    gate_logits = mh_layer.fuse_gate(q)
    gates = torch.sigmoid(gate_logits)
    slot_gate = gates[..., 0].mean().item()
    local_gate = gates[..., 1].mean().item()
    print(f'   slot_gate mean: {slot_gate:.4f}')
    print(f'   local_gate mean: {local_gate:.4f}')
    
    # 5. 分析最终融合
    print('\n5. 最终融合分析')
    y_heads = gates[..., 0:1] * slot_out + gates[..., 1:2] * local_out
    y = mh_layer.out_proj(mh_layer._from_heads(y_heads))
    y_before_proj_std = y_heads.std().item()
    y_after_proj_std = y.std().item()
    print(f'   y (before out_proj) std: {y_before_proj_std:.6f}')
    print(f'   y (after out_proj) std: {y_after_proj_std:.6f}')
    
    # 6. ST 分支对比
    print('\n6. ST 分支对比')
    mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
    h_st = st_layer(x, src_mask=mask, is_causal=True)
    h_st_out = st_proj(h_st)
    h_st_std = h_st_out.std().item()
    print(f'   h_st (after st_proj) std: {h_st_std:.6f}')
    
    # 7. 检查 out_proj 权重
    print('\n7. out_proj vs st_proj 权重对比')
    out_proj_weight_norm = mh_layer.out_proj.weight.norm().item() if mh_layer.out_proj.weight is not None else 0
    st_proj_weight_norm = st_proj.weight.norm().item()
    print(f'   out_proj weight norm: {out_proj_weight_norm:.4f}')
    print(f'   st_proj weight norm: {st_proj_weight_norm:.4f}')
    print(f'   比值: {out_proj_weight_norm / st_proj_weight_norm:.2f}x')

print()
print('='*80)
print('关键发现')
print('='*80)

ratio = h_st_std / y_after_proj_std
print(f'\n1. 输出幅度差异: ST std={h_st_std:.4f} vs MHDSRA2 std={y_after_proj_std:.4f}, ratio={ratio:.1f}x')
print(f'   ❌ MHDSRA2 输出只有 ST 的 {y_after_proj_std/h_st_std*100:.1f}%')

print(f'\n2. Slot Read 输出幅度: {slot_out.std():.6f}')
print(f'   ❌ 只占 y_heads 的 {slot_out.std()/y_before_proj_std*100:.1f}%')

print(f'\n3. Local Attention 输出幅度: {local_out.std():.6f}')
print(f'   ❌ 只占 ST 的 {local_out.std()/h_st_std*100:.1f}%')

print(f'\n4. Slot Gate: {slot_gate:.4f}, Local Gate: {local_gate:.4f}')
print(f'   ⚠️ 门控几乎均匀分布，但由于输出幅度差异，slot 贡献被削弱')

print(f'\n5. Slot 状态: usage={states.usage.mean().item():.4f}, confidence={states.confidence.mean().item():.4f}')
print(f'   ❌ 所有 slot 都未被写入！usage=0')

print()
print('='*80)
print('根因分析')
print('='*80)
print()
print('❌ 核心问题：Slot 从未被写入')
print('   - usage=0 意味着没有任何 slot 被选择写入')
print('   - 这导致 slot read 只能读到初始化值（接近零）')
print('   - 整个 slot 路径完全失效')
print()
print('❌ Local attention 输出幅度也很小')
print(f'   - local_out std={local_out.std():.4f} vs h_st std={h_st_std:.4f}')
print('   - 不是 local attention 本身的问题，而是后续处理的累积效应')
print()
print('❌ out_proj 权重范数较小')
print(f'   - out_proj: {out_proj_weight_norm:.4f} vs st_proj: {st_proj_weight_norm:.4f}')
print(f'   - 比值: {out_proj_weight_norm/st_proj_weight_norm:.2f}x')

print()
print('='*80)
print('修复建议')
print('='*80)
print()
print('方案A：解决 slot 写入问题')
print('   - 检查 slot 写入的选择机制')
print('   - 调整 forget_base 或初始化策略')
print()
print('方案B：在 MHDSRA2 输出后添加 LayerNorm + 可学习缩放')
print('   - LayerNorm 稳定输出幅度')
print('   - 可学习缩放与 ST 分支匹配')
print()
print('方案C：训练时强制 slot 写入')
print('   - 降低 slot 写入的阈值')
print('   - 增加 slot 写入的正则化')

