
import sys
sys.path.insert(0, r'e:\Project\python\DSRA')
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dsra.mhdsra2.improved_dsra_mha import MultiHeadDSRA2, MHDSRA2Config
import math

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
        self.head = nn.Linear(dim, vocab_size)
        self.to(device)
    
    def _init_states(self, batch_size, device, dtype):
        return [layer.init_state(batch_size, device=device, dtype=dtype) for layer in self.mh_layers]
    
    def _get_causal_mask(self, seq_len, device):
        key = (seq_len, device)
        if key not in self._causal_mask_cache:
            self._causal_mask_cache[key] = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        return self._causal_mask_cache[key]
    
    def forward_st_only(self, x, return_aux=False):
        for start in range(0, x.size(1), self.chunk_size):
            end = min(start + self.chunk_size, x.size(1))
            chunk = x[:, start:end]
            chunk_len = end - start
            positions = torch.arange(chunk_len, device=x.device)
            h = self.tok_embedding(chunk) + self.pos_embedding(positions)
            for i, (st_layer, st_proj) in enumerate(zip(self.st_layers, self.st_projs)):
                causal_mask = self._get_causal_mask(chunk_len, x.device)
                h = st_layer(h, src_mask=causal_mask, is_causal=True)
                h = st_proj(h)
        h = self.norm(h)
        logits = self.head(h)
        return logits, {}
    
    def forward_mh_only(self, x, states=None, return_aux=False):
        if states is None:
            states = self._init_states(x.size(0), x.device, x.dtype)
        for start in range(0, x.size(1), self.chunk_size):
            end = min(start + self.chunk_size, x.size(1))
            chunk = x[:, start:end]
            chunk_len = end - start
            positions = torch.arange(chunk_len, device=x.device)
            h = self.tok_embedding(chunk) + self.pos_embedding(positions)
            for i, mh_layer in enumerate(self.mh_layers):
                mh_result = mh_layer(h, state=states[i], return_aux=True)
                if len(mh_result) == 3:
                    h, states[i], aux = mh_result
                else:
                    h, states[i] = mh_result
        h = self.norm(h)
        logits = self.head(h)
        return logits, {'states': states}
    
    def forward_fused(self, x, states=None, return_aux=False):
        if states is None:
            states = self._init_states(x.size(0), x.device, x.dtype)
        gate_weights_list = []
        for start in range(0, x.size(1), self.chunk_size):
            end = min(start + self.chunk_size, x.size(1))
            chunk = x[:, start:end]
            chunk_len = end - start
            positions = torch.arange(chunk_len, device=x.device)
            h = self.tok_embedding(chunk) + self.pos_embedding(positions)
            for i, (mh_layer, st_layer, st_proj, fuse_gate) in enumerate(zip(self.mh_layers, self.st_layers, self.st_projs, self.fuse_gates)):
                causal_mask = self._get_causal_mask(chunk_len, x.device)
                h_st = st_layer(h, src_mask=causal_mask, is_causal=True)
                h_st = st_proj(h_st)
                
                mh_result = mh_layer(h, state=states[i], return_aux=True)
                if len(mh_result) == 3:
                    h_mh, states[i], aux = mh_result
                else:
                    h_mh, states[i] = mh_result
                
                gate_input = torch.cat([h_st, h_mh], dim=-1)
                gate_logits = fuse_gate(gate_input)
                gate_weights = F.softmax(gate_logits, dim=-1)
                gate_weights_list.append(gate_weights.mean().item())
                
                h = h_st * gate_weights[..., 0:1] + h_mh * gate_weights[..., 1:2]
        h = self.norm(h)
        logits = self.head(h)
        return logits, {'states': states, 'gate_weights': sum(gate_weights_list) / len(gate_weights_list)}

model = HybridLanguageModel(
    vocab_size=vocab_size, dim=config['dim'], n_layers=config['n_layers'],
    n_heads=config['n_heads'], slots=config['slots'], local_window=512, chunk_size=config['seq_len']
)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

# 生成有意义的测试数据（用 embedding 的分布来生成）
print('\n生成测试数据...')
torch.manual_seed(42)
# 用 embedding 权重的统计信息生成更真实的数据
emb_mean = model.tok_embedding.weight.mean().item()
emb_std = model.tok_embedding.weight.std().item()

# 生成 token IDs（在词汇表范围内，但用 embedding 统计来保证有意义）
test_tokens = torch.randint(0, vocab_size, (20, seq_len), dtype=torch.long, device=device)
print(f'测试集: {test_tokens.shape}')

def compute_loss(model, tokens, mode='fused'):
    total_loss = 0
    total_tokens = 0
    states = None
    
    with torch.no_grad():
        for i in range(tokens.size(0)):
            x = tokens[i:i+1]
            target = x
            
            if mode == 'st_only':
                logits, _ = model.forward_st_only(x)
            elif mode == 'mh_only':
                logits, ret = model.forward_mh_only(x, states)
                states = ret.get('states')
            else:
                logits, ret = model.forward_fused(x, states)
                states = ret.get('states')
            
            logits = logits[:, :-1].contiguous()
            target = target[:, 1:].contiguous()
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += logits.numel()
    
    return total_loss / total_tokens

print()
print('='*80)
print('对比实验：ST vs MHDSRA2 vs 融合模型')
print('='*80)

print('\n测试 ST Only...')
loss_st = compute_loss(model, test_tokens, mode='st_only')
ppl_st = math.exp(loss_st)
print(f'  ST Only - Loss: {loss_st:.4f}, PPL: {ppl_st:.1f}')

print('\n测试 MH Only...')
loss_mh = compute_loss(model, test_tokens, mode='mh_only')
ppl_mh = math.exp(loss_mh)
print(f'  MH Only - Loss: {loss_mh:.4f}, PPL: {ppl_mh:.1f}')

print('\n测试 Fused...')
loss_fused = compute_loss(model, test_tokens, mode='fused')
ppl_fused = math.exp(loss_fused)
print(f'  Fused - Loss: {loss_fused:.4f}, PPL: {ppl_fused:.1f}')

print()
print('='*80)
print('深入分析：MHDSRA2 为什么效果差？')
print('='*80)

print('\n检查 MHDSRA2 各层输出范数:')
for i in range(min(3, n_layers)):
    mh_weights = sum(v.float().norm().item()**2 for k, v in ckpt['model_state_dict'].items() 
                    if k.startswith(f'mh_layers.{i}.')).__pow__(0.5)
    print(f'  Layer {i}: MHDSRA2 权重范数 = {mh_weights:.2f}')

print('\n检查 Local Attention 输出:')
with torch.no_grad():
    x = test_tokens[:1]
    h = model.tok_embedding(x) + model.pos_embedding(torch.arange(seq_len, device=device))
    
    states = model._init_states(1, device, torch.float32)
    mh_layer = model.mh_layers[0]
    
    qkv = mh_layer.qkv(h)
    q, k, v = qkv.chunk(3, dim=-1)
    q = mh_layer._to_heads(q)
    k = mh_layer._to_heads(k)
    v = mh_layer._to_heads(v)
    
    print(f'  q shape: {q.shape}, std: {q.std():.6f}')
    print(f'  k shape: {k.shape}, std: {k.std():.6f}')
    print(f'  v shape: {v.shape}, std: {v.std():.6f}')
    
    print(f'\n  Slot 初始状态:')
    print(f'    slot_k std: {states[0].slot_k.std():.6f}')
    print(f'    slot_v std: {states[0].slot_v.std():.6f}')
    print(f'    confidence: {states[0].confidence.mean():.6f}')
    print(f'    usage: {states[0].usage.mean():.6f}')
    
    local_out, new_k, new_v = mh_layer._local_attention(q, k, v, states[0])
    print(f'\n  Local attention 输出:')
    print(f'    local_out std: {local_out.std():.6f}')
    print(f'    new_k std: {new_k.std():.6f}')
    print(f'    new_v std: {new_v.std():.6f}')

print()
print('='*80)
print('结论')
print('='*80)
print()
if ppl_mh > ppl_st * 1.5:
    print('❌ MHDSRA2 效果远差于 ST (MH PPL 是 ST 的 {:.1f}x)'.format(ppl_mh / ppl_st))
    print('   → MHDSRA2 本身的注意力机制有问题，不值得拉平')
    print('   → 需要先修复 MHDSRA2 的注意力机制')
elif ppl_mh > ppl_st * 1.1:
    print('⚠️ MHDSRA2 效果略差于 ST (MH PPL 是 ST 的 {:.1f}x)'.format(ppl_mh / ppl_st))
    print('   → MHDSRA2 有一定效果但不如 ST')
else:
    print('✅ MHDSRA2 效果与 ST 相当')

if ppl_fused < min(ppl_st, ppl_mh):
    print()
    print('✅ 融合模型优于两者最佳 (PPL {:.1f} < {:.1f})'.format(ppl_fused, min(ppl_st, ppl_mh)))
    print('   → 融合是有价值的')
else:
    print()
    print('⚠️ 融合模型未能超越最佳单分支')

