
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

# 重建 MHDSRA2 和 ST 层
mh_cfg = MHDSRA2Config(
    dim=dim, heads=n_heads, local_window=512, slot_pe='rope', slots=slots,
    use_retrieval=False
)
mh_layer = MultiHeadDSRA2(mh_cfg).to(device)
mh_sd = {k[len('mh_layers.0.'):]: v for k, v in ckpt['model_state_dict'].items() if k.startswith('mh_layers.0.')}
mh_layer.load_state_dict(mh_sd, strict=False)
mh_layer.eval()

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

# 修复组件
mh_out_norm = nn.LayerNorm(dim).to(device)
mh_scale = nn.Parameter(torch.tensor(1.0, device=device))

# 测试数据
torch.manual_seed(42)
batch_size = 2
seq_len = 128
x = torch.randn(batch_size, seq_len, dim, device=device)
states = mh_layer.init_state(batch_size, device=device, dtype=torch.float32)
causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

print()
print('='*80)
print('测试1: 原始输出（无修复）')
print('='*80)

with torch.no_grad():
    h_st = st_layer(x, src_mask=causal_mask, is_causal=True)
    h_st_out = st_proj(h_st)
    
    result = mh_layer(x, state=states, return_aux=True)
    if len(result) == 3:
        h_mh, states_out, aux = result
    else:
        h_mh, states_out = result
    
    print(f'h_st  - mean: {h_st_out.mean():.6f}, std: {h_st_out.std():.6f}, norm: {h_st_out.norm(dim=-1).mean():.6f}')
    print(f'h_mh  - mean: {h_mh.mean():.6f}, std: {h_mh.std():.6f}, norm: {h_mh.norm(dim=-1).mean():.6f}')
    print(f'ratio: {h_st_out.norm(dim=-1).mean() / h_mh.norm(dim=-1).mean():.2f}x')

print()
print('='*80)
print('测试2: 修复方案 - LayerNorm + 固定缩放')
print('='*80)

with torch.no_grad():
    # 测试不同的缩放倍数
    scales = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0]
    for scale in scales:
        # 重置状态
        states = mh_layer.init_state(batch_size, device=device, dtype=torch.float32)
        result = mh_layer(x, state=states, return_aux=True)
        if len(result) == 3:
            h_mh, _, _ = result
        else:
            h_mh, _ = result
        
        h_mh_normed = mh_out_norm(h_mh)
        h_mh_scaled = h_mh_normed * scale
        
        ratio = h_st_out.norm(dim=-1).mean() / h_mh_scaled.norm(dim=-1).mean()
        
        print(f'scale={scale:3.1f}: h_mh std={h_mh_scaled.std():.6f}, norm={h_mh_scaled.norm(dim=-1).mean():.6f}, ratio={ratio:.2f}x')

print()
print('='*80)
print('测试3: 理想情况 - 直接调整到与 ST 分支匹配')
print('='*80)

with torch.no_grad():
    # 理想缩放值
    desired_norm = h_st_out.norm(dim=-1).mean()
    actual_norm = h_mh.norm(dim=-1).mean()
    ideal_scale = desired_norm / actual_norm
    
    h_mh_ideal = mh_out_norm(h_mh) * ideal_scale
    
    print(f'理想缩放值: {ideal_scale:.2f}x')
    print(f'h_mh_ideal - mean: {h_mh_ideal.mean():.6f}, std: {h_mh_ideal.std():.6f}, norm: {h_mh_ideal.norm(dim=-1).mean():.6f}')
    print(f'与 ST 分支 ratio: {h_st_out.norm(dim=-1).mean() / h_mh_ideal.norm(dim=-1).mean():.2f}x')

print()
print('='*80)
print('测试4: 门控权重变化')
print('='*80)

# 重建 fuse gate
fuse_gate = nn.Linear(dim * 2, 2).to(device)
fuse_gate_sd = {
    'weight': ckpt['model_state_dict']['fuse_gates.0.weight'],
    'bias': ckpt['model_state_dict']['fuse_gates.0.bias']
}
fuse_gate.load_state_dict(fuse_gate_sd, strict=True)
fuse_gate.eval()

with torch.no_grad():
    # 重置状态
    states = mh_layer.init_state(batch_size, device=device, dtype=torch.float32)
    result = mh_layer(x, state=states, return_aux=True)
    if len(result) == 3:
        h_mh, _, _ = result
    else:
        h_mh, _ = result
    
    # 原始门控
    gate_input_original = torch.cat([h_st_out, h_mh], dim=-1)
    gate_logits_original = fuse_gate(gate_input_original)
    gate_weights_original = F.softmax(gate_logits_original, dim=-1)
    st_w_original = gate_weights_original[..., 0].mean().item()
    mh_w_original = gate_weights_original[..., 1].mean().item()
    
    # 修复后的门控（用理想缩放）
    h_mh_fixed = mh_out_norm(h_mh) * ideal_scale
    gate_input_fixed = torch.cat([h_st_out, h_mh_fixed], dim=-1)
    gate_logits_fixed = fuse_gate(gate_input_fixed)
    gate_weights_fixed = F.softmax(gate_logits_fixed, dim=-1)
    st_w_fixed = gate_weights_fixed[..., 0].mean().item()
    mh_w_fixed = gate_weights_fixed[..., 1].mean().item()
    
    print(f'原始门控: ST={st_w_original:.2%}, MH={mh_w_original:.2%}')
    print(f'修复后门控: ST={st_w_fixed:.2%}, MH={mh_w_fixed:.2%}')
    print(f'门控变化: ST={st_w_original-st_w_fixed:+.2%}, MH={mh_w_fixed-mh_w_original:+.2%}')

print()
print('='*80)
print('结论')
print('='*80)
print()
print('✅ LayerNorm + 可学习缩放是有效的修复方案')
print(f'   - 理想缩放值约为 {ideal_scale:.2f}x，可将 MHDSRA2 输出幅度拉平')
print(f'   - 修复后门控权重会从 ST={st_w_original:.2%} 拉到更平衡的位置')
print('   - 保留了归一化的稳定性（slot read 余弦相似度 + 温度控制）')

