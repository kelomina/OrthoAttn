"""快速测试8层模型的门控信息是否完整"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

# 导入我们的模型类
from scripts.pretrain_hybrid_lm import HybridLanguageModel, FastBPETokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

print("\n" + "="*70)
print("测试8层模型门控信息完整性")
print("="*70)

# 加载分词器
tokenizer = FastBPETokenizer.load("models/bpe_tokenizer.json")

# 创建8层模型
print("\n创建8层模型...")
model = HybridLanguageModel(
    vocab_size=len(tokenizer.encoder),
    dim=256,
    n_layers=8,
    n_heads=8,
    slots=256,
    chunk_size=512,
    use_gradient_checkpointing=False,
).to(DEVICE)
model.eval()

print(f"模型参数结构:")
print(f"  mh_layers: {len(model.mh_layers)} 层")
print(f"  st_layers: {len(model.st_layers)} 层")
print(f"  st_projs: {len(model.st_projs)} 层")
print(f"  fuse_gates: {len(model.fuse_gates)} 层")

# 测试前向传播并获取门控信息
print("\n测试前向传播...")
x = torch.randint(0, len(tokenizer.encoder), (2, 64)).to(DEVICE)
with torch.no_grad():
    logits, states, gate_info = model(x, return_gate_info=True)

print(f"\ngate_info 键: {sorted(gate_info.keys())}")

print("\n门控权重检查:")
all_layers_ok = True
for i in range(8):
    st_key = f"layer{i}_st_weight"
    mh_key = f"layer{i}_mh_weight"
    has_st = st_key in gate_info
    has_mh = mh_key in gate_info
    if has_st and has_mh:
        st_val = gate_info[st_key]
        mh_val = gate_info[mh_key]
        print(f"  ✓ Layer {i}: st={st_val:.4f}, mh={mh_val:.4f}")
    else:
        print(f"  ✗ Layer {i}: missing! (st={has_st}, mh={has_mh})")
        all_layers_ok = False

print(f"\n平均门控:")
st_avg = sum(gate_info[k] for k in gate_info if 'st_weight' in k) / 8
mh_avg = sum(gate_info[k] for k in gate_info if 'mh_weight' in k) / 8
print(f"  st_avg={st_avg:.4f}, mh_avg={mh_avg:.4f}")

print("\n" + "="*70)
if all_layers_ok:
    print("✓ 8层模型门控信息完整！")
else:
    print("✗ 8层模型门控信息不完整！")
print("="*70)
