
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
print('修复方案测试：给 MHDSRA2 输出添加 LayerNorm + 可学习缩放')
print('='*80)

with torch.no_grad():
    print()
    print('原始模型：')
    logits, new_states, gate_info = model(x, states=states, return_gate_info=True)
    for i in range(n_layers):
        st_w = gate_info[f"layer{i}_st_weight"]
        mh_w = gate_info[f"layer{i}_mh_weight"]
        print(f'  Layer {i}: ST={st_w:.2%}, MH={mh_w:.2%}')
    
    print()
    print('='*80)
    print('测试修复方案：')
    print('='*80)
    
    # 测试：修改 forward 给每个 h_mh 添加 LayerNorm + 1.0 scale
    print()
    print('方案：对 h_mh 应用 LayerNorm，然后乘以 1.0：')
    with torch.no_grad():
        states = model._init_states(batch_size, device, torch.float32)
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
                
                # 修复方案：给 h_mh 应用 LayerNorm
                h_mh = F.layer_norm(h_mh, h_mh.shape[-1:])
                
                gate_input = torch.cat([h_st, h_mh], dim=-1)
                gate_logits = fuse_gate(gate_input)
                gate_weights = F.softmax(gate_logits, dim=-1)
                
                h = gate_weights[..., 0:1] * h_st + gate_weights[..., 1:2] * h_mh
                
                if i == n_layers - 1:
                    print(f'  Layer {i}:')
                    print(f'    h_st - norm: {h_st.norm(dim=-1).mean().item():.4f}')
                    print(f'    h_mh (LayerNorm) - norm: {h_mh.norm(dim=-1).mean().item():.4f}')
                    print(f'    gate ST: {gate_weights[..., 0].mean().item():.2%}, MH: {gate_weights[..., 1].mean().item():.2%}')
    
    print()
    print('='*80)
    print('建议的完整修复方案：')
    print('='*80)
    print()
    print('1. 在 HybridLanguageModel 中添加：')
    print('   - 每个 MHDSRA2 层对应一个 LayerNorm 层 (self.mh_out_norms)')
    print('   - 每个 MHDSRA2 层对应一个可学习缩放参数 (self.mh_scales)')
    print()
    print('2. 在 forward 中，对 h_mh 应用：')
    print('   h_mh = self.mh_out_norms[i](h_mh)')
    print('   h_mh = h_mh * self.mh_scales[i]')
    print()
    print('3. 可选：增加 gate_reg_weight，强制门控更均匀')

