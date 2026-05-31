
import sys
sys.path.insert(0, r'e:\Project\python\DSRA')
import torch
import torch.nn as nn
import torch.nn.functional as F

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
n_layers = config['n_layers']

# 重建整个融合模型
from scripts.pretrain_hybrid_lm import HybridLanguageModel

model = HybridLanguageModel(
    vocab_size=vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads,
    slots=slots, local_window=512, chunk_size=seq_len
)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
model.to(device)

# 测试数据
torch.manual_seed(42)
batch_size = 2
x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
states = model._init_states(batch_size, device, torch.float32)

print()
print('='*80)
print('融合模型调试')
print('='*80)

with torch.no_grad():
    # 做一次正向传播
    logits, new_states, gate_info = model(x, states=states, return_gate_info=True)
    
    print()
    print('门控权重:')
    for i in range(n_layers):
        st_w = gate_info[f"layer{i}_st_weight"]
        mh_w = gate_info[f"layer{i}_mh_weight"]
        print(f'  Layer {i}: ST={st_w:.2%}, MH={mh_w:.2%}')
    
    # 深入检查每个层的内部
    print()
    print('='*80)
    print('逐层调试')
    print('='*80)
    
    with torch.no_grad():
        for start in range(0, x.size(1), model.chunk_size):
            end = min(start + model.chunk_size, x.size(1))
            chunk = x[:, start:end]
            chunk_len = end - start
            positions = torch.arange(chunk_len, device=x.device)
            h = model.tok_embedding(chunk) + model.pos_embedding(positions)
            
            for i, (mh_layer, st_layer, st_proj, fuse_gate) in enumerate(
                zip(model.mh_layers, model.st_layers, model.st_projs, model.fuse_gates)
            ):
                causal_mask = model._get_causal_mask(chunk_len, x.device)
                h_st = st_layer(h, src_mask=causal_mask, is_causal=True)
                h_st = st_proj(h_st)
                
                mh_result = mh_layer(h, state=states[i], return_aux=True)
                if len(mh_result) == 3:
                    h_mh, states[i], _ = mh_result
                else:
                    h_mh, states[i] = mh_result
                
                gate_input = torch.cat([h_st, h_mh], dim=-1)
                gate_logits = fuse_gate(gate_input)
                gate_weights = F.softmax(gate_logits, dim=-1)
                
                h = gate_weights[..., 0:1] * h_st + gate_weights[..., 1:2] * h_mh
                
                if i == n_layers - 1:  # 只看最后一层
                    print()
                    print(f'Layer {i}:')
                    print(f'  h_st - mean: {h_st.mean().item():.4f}, std: {h_st.std().item():.4f}, norm: {h_st.norm(dim=-1).mean().item():.4f}')
                    print(f'  h_mh - mean: {h_mh.mean().item():.4f}, std: {h_mh.std().item():.4f}, norm: {h_mh.norm(dim=-1).mean().item():.4f}')
                    print(f'  h_st/h_mh norm ratio: {h_st.norm(dim=-1).mean().item() / h_mh.norm(dim=-1).mean().item():.2f}x')
                    print(f'  gate ST: {gate_weights[..., 0].mean().item():.2%}, MH: {gate_weights[..., 1].mean().item():.2%}')

print()
print('='*80)
print('检查 fuse_gate 的权重')
print('='*80)

with torch.no_grad():
    for i in range(n_layers):
        fg = model.fuse_gates[i]
        print()
        print(f'Layer {i} fuse_gate:')
        print(f'  weight shape: {fg.weight.shape}')
        print(f'  weight norm: {fg.weight.norm().item():.4f}')
        print(f'  bias: {fg.bias.data.tolist()}')
        
        # 分割权重看 ST 和 MH 的部分
        w_st = fg.weight[:, :dim]
        w_mh = fg.weight[:, dim:]
        print(f'  w_st norm: {w_st.norm().item():.4f}, w_mh norm: {w_mh.norm().item():.4f}')
        print(f'  w_st vs w_mh ratio: {w_st.norm().item() / w_mh.norm().item():.2f}x')

