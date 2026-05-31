import sys
sys.path.insert(0, r'e:\Project\python\DSRA')
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

from src.dsra.mhdsra2.improved_dsra_mha import MultiHeadDSRA2, MHDSRA2Config

ckpt = torch.load(r'e:\Project\python\DSRA\models\hybrid_lm\best_model.pt', map_location=device, weights_only=False)
config = ckpt['config']
vocab_size = ckpt['vocab_size']
print(f'配置: {config}, vocab_size={vocab_size}')

class HybridLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, slots, local_window, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.dim = dim
        self.n_layers = n_layers
        self._causal_mask_cache = {}
        self.tok_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(chunk_size, dim)
        mh_cfg = MHDSRA2Config(
            dim=dim, heads=n_heads, local_window=local_window,
            slot_pe='rope', slots=slots, tau_init=8.0, tau_write_init=4.0,
            read_topk=8, write_topk=4, use_retrieval=False,
            forget_base=0.001, usage_decay=0.995, conf_decay=0.999,
        )
        self.mh_layers = nn.ModuleList([MultiHeadDSRA2(mh_cfg) for _ in range(n_layers)])
        self.st_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim*4,
                                        batch_first=True, activation='gelu') for _ in range(n_layers)
        ])
        self.st_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.fuse_gates = nn.ModuleList([nn.Linear(dim * 2, 2) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.to(device)
    
    def _init_states(self, batch_size, device, dtype):
        return [layer.init_state(batch_size, device=device, dtype=dtype) for layer in self.mh_layers]
    
    def _get_causal_mask(self, seq_len, device):
        key = (seq_len, device)
        if key not in self._causal_mask_cache:
            self._causal_mask_cache[key] = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        return self._causal_mask_cache[key]

model = HybridLanguageModel(
    vocab_size=vocab_size, dim=config['dim'], n_layers=config['n_layers'],
    n_heads=config['n_heads'], slots=config['slots'], local_window=512, chunk_size=config['seq_len']
)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
print('模型加载成功')

torch.manual_seed(42)
batch_size = 2
seq_len = config['seq_len']
x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

states = model._init_states(batch_size, device, torch.float32)

print()
print('===== 逐层门控权重和分支输出分析 =====')
with torch.no_grad():
    for start in range(0, seq_len, model.chunk_size):
        end = min(start + model.chunk_size, seq_len)
        chunk = x[:, start:end]
        chunk_len = end - start
        positions = torch.arange(chunk_len, device=device)
        h = model.tok_embedding(chunk) + model.pos_embedding(positions)
        
        for i, (mh_layer, st_layer, st_proj, fuse_gate) in enumerate(
            zip(model.mh_layers, model.st_layers, model.st_projs, model.fuse_gates)
        ):
            causal_mask = model._get_causal_mask(chunk_len, device)
            h_st = st_layer(h, src_mask=causal_mask, is_causal=True)
            h_st_out = st_proj(h_st)
            
            mh_result = mh_layer(h, state=states[i], return_aux=True)
            if len(mh_result) == 3:
                h_mh, states[i], aux = mh_result
            else:
                h_mh, states[i] = mh_result
            
            gate_input = torch.cat([h_st_out, h_mh], dim=-1)
            gate_logits = fuse_gate(gate_input)
            gate_weights = F.softmax(gate_logits, dim=-1)
            
            st_w = gate_weights[..., 0].mean().item()
            mh_w = gate_weights[..., 1].mean().item()
            st_std = gate_weights[..., 0].std().item()
            
            h_st_norm = h_st_out.norm(dim=-1).mean().item()
            h_mh_norm = h_mh.norm(dim=-1).mean().item()
            h_st_std = h_st_out.std().item()
            h_mh_std = h_mh.std().item()
            
            has_nan_mh = torch.isnan(h_mh).any().item()
            has_inf_mh = torch.isinf(h_mh).any().item()
            
            print(f'  Layer {i}: ST_w={st_w:.4f}±{st_std:.4f}, MH_w={mh_w:.4f}')
            print(f'    h_st: norm={h_st_norm:.4f}, std={h_st_std:.6f}')
            print(f'    h_mh: norm={h_mh_norm:.4f}, std={h_mh_std:.6f}, nan={has_nan_mh}, inf={has_inf_mh}')

print()
print('===== Slot 记忆状态 =====')
for i, state in enumerate(states):
    conf = state.confidence
    usage = state.usage
    slot_k = state.slot_k
    print(f'  Layer {i}:')
    print(f'    confidence: mean={conf.mean().item():.4f}, min={conf.min().item():.4f}, max={conf.max().item():.4f}')
    print(f'    usage: mean={usage.mean().item():.4f}, min={usage.min().item():.4f}, max={usage.max().item():.4f}')
    slot_k_flat = slot_k.reshape(-1, slot_k.shape[-1])
    cos_sim = F.cosine_similarity(slot_k_flat.unsqueeze(0), slot_k_flat.unsqueeze(1), dim=-1)
    off_diag = cos_sim[~torch.eye(cos_sim.shape[0], dtype=bool, device=device)]
    print(f'    slot_k avg_cosine_sim={off_diag.mean().item():.4f} (越低越多样)')
    print(f'    position={state.position}')

print()
print('===== 诊断完成 =====')
