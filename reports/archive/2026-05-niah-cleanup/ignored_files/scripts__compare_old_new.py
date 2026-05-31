"""
交叉验证：对比新旧配置的 conflict 和 write_gate
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from scripts.pretrain_hybrid_lm import HybridLanguageModel

# 测试1：新配置的模型
print("=" * 80)
print("新配置模型（read_topk=2, hard_read=True, eta=1.0, max_update=1.0）")
print("=" * 80)

torch.manual_seed(42)
model_new = HybridLanguageModel(
    vocab_size=1000, dim=256, n_layers=8, n_heads=8,
    slots=256, chunk_size=512
).to(DEVICE)

batch_size = 2
seq_len = 512
x_tokens = torch.randint(0, 1000, (batch_size, seq_len), device=DEVICE)

with torch.no_grad():
    logits, states, gate_info = model_new(x_tokens, return_gate_info=True)
    
    print("Gate weights:")
    for i in range(8):
        st_w = gate_info.get(f"layer{i}_st_weight", 0)
        mh_w = gate_info.get(f"layer{i}_mh_weight", 0)
        tg = gate_info.get(f"layer{i}_token_gate_mean", 0)
        usage = gate_info.get(f"layer{i}_usage_mean", 0)
        print(f"  Layer {i}: ST={st_w:.4f}, MH={mh_w:.4f}, token_gate={tg:.4f}, usage={usage:.4f}")

# 测试2：检查新配置模型的 conflict
print()
print("检查新配置模型的 conflict（通过 forward 获取）:")
with torch.no_grad():
    for i in range(8):
        mh = model_new.mh_layers[i]
        states_i = mh.init_state(batch_size, device=DEVICE, dtype=torch.float32)
        
        positions = torch.arange(seq_len, device=DEVICE)
        h = model_new.tok_embedding(x_tokens) + model_new.pos_embedding(positions)
        
        _, states_after, aux = mh(h[:1], state=states_i, return_aux=True)
        ws = aux.get("write_stats", {})
        
        conflict = ws.get("conflict_mean", torch.tensor(0)).item()
        write_gate = ws.get("write_gate_mean", torch.tensor(0)).item()
        forget_gate = ws.get("forget_gate_mean", torch.tensor(0)).item()
        
        print(f"  Layer {i}: conflict={conflict:.4f}, write_gate={write_gate:.4f}, forget_gate={forget_gate:.4f}")

# 测试3：加载旧检查点，对比
print()
print("=" * 80)
print("旧检查点模型（对比）")
print("=" * 80)

ckpt_path = PROJECT_ROOT / "models" / "hybrid_lm" / "best_model.pt"
ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
config = ckpt['config']
vocab_size = ckpt['vocab_size']

model_old = HybridLanguageModel(
    vocab_size=vocab_size, dim=config['dim'], n_layers=config['n_layers'],
    n_heads=config['n_heads'], slots=config['slots'],
    chunk_size=config.get('seq_len', 512)
).to(DEVICE)
model_old.load_state_dict(ckpt['model_state_dict'], strict=False)
model_old.eval()

x_tokens_old = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)

with torch.no_grad():
    for i in range(8):
        mh = model_old.mh_layers[i]
        states_i = mh.init_state(batch_size, device=DEVICE, dtype=torch.float32)
        
        positions = torch.arange(seq_len, device=DEVICE)
        h = model_old.tok_embedding(x_tokens_old) + model_old.pos_embedding(positions)
        
        _, states_after, aux = mh(h[:1], state=states_i, return_aux=True)
        ws = aux.get("write_stats", {})
        
        conflict = ws.get("conflict_mean", torch.tensor(0)).item()
        write_gate = ws.get("write_gate_mean", torch.tensor(0)).item()
        forget_gate = ws.get("forget_gate_mean", torch.tensor(0)).item()
        
        print(f"  Layer {i}: conflict={conflict:.4f}, write_gate={write_gate:.4f}, forget_gate={forget_gate:.4f}")

print()
print("关键对比：")
print(f"旧模型 config: read_topk={model_old.mh_layers[0].cfg.read_topk}, hard_read={model_old.mh_layers[0].cfg.hard_read}, eta={model_old.mh_layers[0].cfg.eta}, max_update={model_old.mh_layers[0].cfg.max_update}")
print(f"新模型 config: read_topk={model_new.mh_layers[0].cfg.read_topk}, hard_read={model_new.mh_layers[0].cfg.hard_read}, eta={model_new.mh_layers[0].cfg.eta}, max_update={model_new.mh_layers[0].cfg.max_update}")
