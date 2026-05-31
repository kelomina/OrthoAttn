"""诊断模型门控数据缺失问题"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from scripts.pretrain_hybrid_lm import HybridLanguageModel, FastBPETokenizer
from src.dsra.mhdsra2.improved_dsra_mha import MultiHeadDSRA2, MHDSRA2Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试1: 创建8层模型，检查forward返回的gate_info
print("=" * 70)
print("测试1: 检查8层模型前向传播")
print("=" * 70)

tokenizer = FastBPETokenizer.load("models/bpe_tokenizer.json")
model = HybridLanguageModel(
    vocab_size=len(tokenizer.encoder),
    dim=256, n_layers=8, n_heads=8,
    slots=256, chunk_size=512,
    use_gradient_checkpointing=False,
).to(DEVICE)

print(f"模型层数: model.n_layers = {model.n_layers}")
print(f"mh_layers 长度: {len(model.mh_layers)}")
print(f"st_layers 长度: {len(model.st_layers)}")
print(f"st_projs 长度: {len(model.st_projs)}")
print(f"fuse_gates 长度: {len(model.fuse_gates)}")

x = torch.randint(0, len(tokenizer.encoder), (2, 128)).to(DEVICE)
logits, states, gate_info = model(x, return_gate_info=True)

print(f"\ngate_info 包含的键:")
for key in sorted(gate_info.keys()):
    print(f"  {key} = {gate_info[key]:.4f}")

# 检查是否所有8层都有数据
for i in range(8):
    st_key = f"layer{i}_st_weight"
    mh_key = f"layer{i}_mh_weight"
    if st_key in gate_info and mh_key in gate_info:
        print(f"✓ Layer {i} 正常: st={gate_info[st_key]:.4f}, mh={gate_info[mh_key]:.4f}")
    else:
        print(f"✗ Layer {i} 缺失")

print("\n" + "=" * 70)
print("测试2: 检查zip循环是否截断")
print("=" * 70)
layers_zip = list(zip(model.mh_layers, model.st_layers, model.st_projs, model.fuse_gates))
print(f"zip 循环长度: {len(layers_zip)}")
for i, (mh, st, proj, gate) in enumerate(layers_zip):
    print(f"  zip 第{i}项: mh={type(mh).__name__}, st={type(st).__name__}, proj={type(proj).__name__}, gate={type(gate).__name__}")

print("\n" + "=" * 70)
print("测试3: 检查states初始化")
print("=" * 70)
states = model._init_states(2, DEVICE, torch.float32)
print(f"states 长度: {len(states)}")

print("\n诊断完成!")
