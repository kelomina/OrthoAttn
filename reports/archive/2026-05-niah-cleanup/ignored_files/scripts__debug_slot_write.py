
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

print()
print('='*80)
print('Slot 写入机制完整调试')
print('='*80)

with torch.no_grad():
    print('初始状态:')
    print(f'  usage mean: {states.usage.mean().item():.4f}')
    print(f'  confidence mean: {states.confidence.mean().item():.4f}')
    print(f'  age mean: {states.age.mean().item():.4f}')
    
    y, new_states, aux = mh_layer(x, state=states, return_aux=True)
    
    print()
    print('一次 forward 后的状态:')
    print(f'  usage mean: {new_states.usage.mean().item():.4f}')
    print(f'  usage max: {new_states.usage.max().item():.4f}')
    print(f'  confidence mean: {new_states.confidence.mean().item():.4f}')
    print(f'  age mean: {new_states.age.mean().item():.4f}')
    
    # 调试 aux 数据
    if aux is not None:
        print()
        print('Aux 信息:')
        for k, v in aux.items():
            if isinstance(v, torch.Tensor):
                print(f'  {k}: shape={v.shape}, mean={v.mean().item():.6f}, max={v.max().item():.6f}')
            else:
                print(f'  {k}: {v}')
    
    # 手动模拟一次完整的 slot write
    print()
    print('='*80)
    print('手动模拟 slot write')
    print('='*80)
    
    qkv = mh_layer.qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)
    q = mh_layer._to_heads(q)
    k = mh_layer._to_heads(k)
    v = mh_layer._to_heads(v)
    
    # 1. 先做一次 slot read 来获得 read_mass
    print()
    print('1. Slot read (获取 read_mass)')
    slot_out, read_aux = mh_layer._slot_read(q, new_states)  # 用新的 states
    if read_aux and 'read_mass' in read_aux:
        read_mass = read_aux['read_mass']
        print(f'   read_mass shape: {read_mass.shape}, mean: {read_mass.mean().item():.6f}, max: {read_mass.max().item():.6f}')
        
        # 2. 现在做 slot write
        print()
        print('2. Slot write')
        cfg = mh_cfg
        batch_size, heads, seq_len, d_head = k.shape
        slot_k = new_states.slot_k.to(dtype=k.dtype)
        slot_v = new_states.slot_v.to(dtype=v.dtype)
        kn = F.normalize(k, dim=-1)
        sk = F.normalize(slot_k, dim=-1)
        sim = torch.einsum("bhtd,bhkd->bhtk", kn, sk)
        max_sim = sim.max(dim=-1).values
        novelty = (1.0 - max_sim).clamp(0.0, 1.0)
        print(f'   novelty mean: {novelty.mean().item():.6f}, max: {novelty.max().item():.6f}')
        
        tau = mh_layer.log_tau_write.exp().to(dtype=k.dtype)
        tau = tau.clamp(min=1.0, max=64.0)
        print(f'   tau_write: {tau.item():.4f}')
        
        read_hint = read_mass.to(dtype=k.dtype).unsqueeze(2).clamp(0.0, 1.0)
        usage_penalty = cfg.usage_prior * torch.log1p(new_states.usage).to(dtype=k.dtype).unsqueeze(2)
        write_logits = sim * tau
        write_logits = write_logits - usage_penalty * (1.0 - read_hint)
        write_logits = write_logits + cfg.age_write_bias * torch.log1p(new_states.age).to(dtype=k.dtype).unsqueeze(2)
        write_logits = write_logits + tau * read_hint
        print(f'   write_logits mean: {write_logits.mean().item():.6f}, max: {write_logits.max().item():.6f}, min: {write_logits.min().item():.6f}')
        
        w_top = min(cfg.write_topk, cfg.slots)
        top_logits, top_idx = torch.topk(write_logits, w_top, dim=-1)
        route = F.softmax(top_logits, dim=-1)
        print(f'   route mean: {route.mean().item():.6f}, max: {route.max().item():.6f}')
        
        token_gate = torch.sigmoid(mh_layer.token_write_gate(k)).squeeze(-1) * novelty
        print(f'   token_write_gate mean: {torch.sigmoid(mh_layer.token_write_gate(k)).mean().item():.6f}')
        print(f'   token_gate mean: {token_gate.mean().item():.6f}, max: {token_gate.max().item():.6f}')
        
        weights = route * token_gate.unsqueeze(-1)
        print(f'   weights mean: {weights.mean().item():.6f}, max: {weights.max().item():.6f}, sum: {weights.sum().item():.4f}')
        
        # 检查有多少 weights 是 > 0 的
        nonzero_weights = (weights > 1e-6).sum().item()
        total_weights = weights.numel()
        print(f'   非零权重数量: {nonzero_weights}/{total_weights} ({nonzero_weights/total_weights*100:.2f}%)')

print()
print('='*80)
print('关键检查')
print('='*80)
print()

with torch.no_grad():
    # 检查 token_write_gate 的输出
    qkv = mh_layer.qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)
    k = mh_layer._to_heads(k)
    tg_out = mh_layer.token_write_gate(k)
    tg_sigmoid = torch.sigmoid(tg_out)
    print(f'token_write_gate output - mean: {tg_out.mean().item():.4f}, std: {tg_out.std().item():.4f}')
    print(f'token_write_gate sigmoid - mean: {tg_sigmoid.mean().item():.6f}, max: {tg_sigmoid.max().item():.4f}')
    
    if tg_sigmoid.max() < 0.1:
        print()
        print('❌ 发现问题！token_write_gate 的输出几乎全是 0！')
        print('   → 这是 slot 从未被写入的原因！')

