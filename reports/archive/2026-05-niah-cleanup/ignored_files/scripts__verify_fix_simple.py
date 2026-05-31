
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

# fuse gate
fuse_gate = nn.Linear(dim * 2, 2).to(device)
fuse_gate_sd = {
    'weight': ckpt['model_state_dict']['fuse_gates.0.weight'],
    'bias': ckpt['model_state_dict']['fuse_gates.0.bias']
}
fuse_gate.load_state_dict(fuse_gate_sd, strict=True)
fuse_gate.eval()

# 测试数据
torch.manual_seed(42)
batch_size = 2
seq_len = 128
x = torch.randn(batch_size, seq_len, dim, device=device)
causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

print()
print('='*80)
print('测试: 更优修复方案 - 直接缩放（不使用 LayerNorm）')
print('='*80)

with torch.no_grad():
    h_st = st_layer(x, src_mask=causal_mask, is_causal=True)
    h_st_out = st_proj(h_st)
    
    # 运行 MHDSRA2
    states = mh_layer.init_state(batch_size, device=device, dtype=torch.float32)
    result = mh_layer(x, state=states, return_aux=True)
    if len(result) == 3:
        h_mh, _, _ = result
    else:
        h_mh, _ = result
    
    # 原始
    gate_input_original = torch.cat([h_st_out, h_mh], dim=-1)
    gate_logits_original = fuse_gate(gate_input_original)
    gate_weights_original = F.softmax(gate_logits_original, dim=-1)
    st_w_original = gate_weights_original[..., 0].mean().item()
    mh_w_original = gate_weights_original[..., 1].mean().item()
    
    print(f'原始门控: ST={st_w_original:.2%}, MH={mh_w_original:.2%}')
    print(f'h_st norm: {h_st_out.norm(dim=-1).mean():.4f}, h_mh norm: {h_mh.norm(dim=-1).mean():.4f}')
    print()
    
    # 测试不同缩放值
    print('测试不同缩放值:')
    scales = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0]
    for scale in scales:
        states = mh_layer.init_state(batch_size, device=device, dtype=torch.float32)
        result = mh_layer(x, state=states, return_aux=True)
        if len(result) == 3:
            h_mh, _, _ = result
        else:
            h_mh, _ = result
        
        h_mh_scaled = h_mh * scale
        
        gate_input = torch.cat([h_st_out, h_mh_scaled], dim=-1)
        gate_logits = fuse_gate(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)
        st_w = gate_weights[..., 0].mean().item()
        mh_w = gate_weights[..., 1].mean().item()
        
        h_ratio = h_st_out.norm(dim=-1).mean() / h_mh_scaled.norm(dim=-1).mean()
        
        print(f'scale={scale:4.1f}: ST={st_w:.2%}, MH={mh_w:.2%}, h_ratio={h_ratio:.2f}x')
    
    print()
    
    # 找到最佳缩放（门控最接近50/50）
    best_scale = None
    best_diff = float('inf')
    best_st_w = None
    best_mh_w = None
    
    for scale in [i/10 for i in range(10, 201)]:
        states = mh_layer.init_state(batch_size, device=device, dtype=torch.float32)
        result = mh_layer(x, state=states, return_aux=True)
        if len(result) == 3:
            h_mh, _, _ = result
        else:
            h_mh, _ = result
        
        h_mh_scaled = h_mh * scale
        gate_input = torch.cat([h_st_out, h_mh_scaled], dim=-1)
        gate_logits = fuse_gate(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)
        st_w = gate_weights[..., 0].mean().item()
        mh_w = gate_weights[..., 1].mean().item()
        
        diff = abs(st_w - 0.5)
        if diff < best_diff:
            best_diff = diff
            best_scale = scale
            best_st_w = st_w
            best_mh_w = mh_w
    
    print(f'最佳缩放: scale={best_scale:.1f}, ST={best_st_w:.2%}, MH={best_mh_w:.2%}')

print()
print('='*80)
print('结论')
print('='*80)
print()
print('✅ 直接添加可学习缩放参数是最简单有效的修复')
print(f'   - 最佳缩放约为 {best_scale:.1f}x，可将门控拉到接近 50/50')
print('   - 方案：在 MultiHeadDSRA2 的 out_proj 后添加 nn.Parameter(1.0)')

