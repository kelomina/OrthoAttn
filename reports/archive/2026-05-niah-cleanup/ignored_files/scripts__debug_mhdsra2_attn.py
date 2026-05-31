
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

# ST 层对比
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

# 运行对比
torch.manual_seed(42)
batch_size = 2
seq_len = 128
x = torch.randn(batch_size, seq_len, dim, device=device)
states = mh_layer.init_state(batch_size, device=device, dtype=torch.float32)

with torch.no_grad():
    # ST 分支（带因果mask）
    mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
    h_st = st_layer(x, src_mask=mask, is_causal=True)
    h_st_out = st_proj(h_st)
    
    # MHDSRA2 分支
    mh_result = mh_layer(x, state=states, return_aux=True)
    if len(mh_result) == 3:
        h_mh, states_out, aux = mh_result
    else:
        h_mh, states_out = mh_result
    
    print()
    print('=' * 80)
    print('ST vs MHDSRA2 输出对比')
    print('=' * 80)
    print(f'h_st  - mean: {h_st.mean():.6f}, std: {h_st.std():.6f}, norm mean: {h_st.norm(dim=-1).mean():.6f}')
    print(f'h_mh  - mean: {h_mh.mean():.6f}, std: {h_mh.std():.6f}, norm mean: {h_mh.norm(dim=-1).mean():.6f}')
    print(f'ratio: {h_st.norm(dim=-1).mean() / h_mh.norm(dim=-1).mean():.2f}x')
    
    print()
    print('=' * 80)
    print('深入分析 MHDSRA2 内部')
    print('=' * 80)
    
    # 重新运行 MHDSRA2，追踪中间状态
    qkv = mh_layer.qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)
    q = mh_layer._to_heads(q)
    k = mh_layer._to_heads(k)
    v = mh_layer._to_heads(v)
    
    print(f'q (after to_heads) - mean: {q.mean():.6f}, std: {q.std():.6f}')
    print(f'k (after to_heads) - mean: {k.mean():.6f}, std: {k.std():.6f}')
    
    # Slot read
    slot_k = states.slot_k.to(dtype=q.dtype)
    slot_v = states.slot_v.to(dtype=q.dtype)
    qn = F.normalize(q, dim=-1)
    sk = F.normalize(slot_k, dim=-1)
    
    print(f'qn (normalized) - mean: {qn.mean():.6f}, std: {qn.std():.6f}')
    print(f'sk (normalized) - mean: {sk.mean():.6f}, std: {sk.std():.6f}')
    
    # Slot read logits
    tau = mh_layer.log_tau_read.exp().to(dtype=q.dtype)
    logits = torch.einsum("bhtd,bhkd->bhtk", qn, sk) * tau
    print(f'logits - mean: {logits.mean():.6f}, std: {logits.std():.6f}')
    
    # Slot read output
    slot_out, aux_read = mh_layer._slot_read(q, states)
    print(f'slot_out - mean: {slot_out.mean():.6f}, std: {slot_out.std():.6f}')
    
    # Local attention
    local_out, new_k, new_v = mh_layer._local_attention(q, k, v, states)
    print(f'local_out - mean: {local_out.mean():.6f}, std: {local_out.std():.6f}')
    
    # MHDSRA2内部门控
    gate_logits = mh_layer.fuse_gate(q)
    gates = torch.sigmoid(gate_logits)
    print(f'gates (slot/local/retr) - mean: {gates.mean(dim=(0,2))}')
    
    # 最终融合
    y_heads = gates[..., 0:1] * slot_out + gates[..., 1:2] * local_out
    y = mh_layer.out_proj(mh_layer._from_heads(y_heads))
    
    print(f'final mhdsra2 y - mean: {y.mean():.6f}, std: {y.std():.6f}')
    
    print()
    print('=' * 80)
    print('核心发现')
    print('=' * 80)
    print('1. Slot read 对 q 和 slot_k 做了 L2 归一化，导致输出幅度被压缩')
    print('2. Local attention 使用 SDPA，输出可能也被压缩')
    print('3. out_proj 可能没有足够的增益来补偿')
    print('4. 对比：ST 分支没有任何归一化，输出幅度自然更大')

