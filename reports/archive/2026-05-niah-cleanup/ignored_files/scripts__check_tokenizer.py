import json
import sys
from pathlib import Path

# 确保可以导入项目
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pretrain_hybrid_lm import FastBPETokenizer

# 检查分词器文件
tokenizer_path = "models/bpe_tokenizer.json"
model_path = "models/hybrid_lm/best_model.pt"

print("="*60)
print("BPE分词器状态检查")
print("="*60)

# 检查分词器文件
if Path(tokenizer_path).exists():
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✓ 分词器文件: {tokenizer_path}")
    print(f"  词汇表大小: {len(data['encoder']):,}")
    print(f"  合并次数: {len(data['merges']):,}")
    print(f"  特殊tokens: {len(data['special_tokens'])}个")
    
    # 加载分词器并测试
    print("\n加载分词器并测试编码效果:")
    tokenizer = FastBPETokenizer.load(tokenizer_path)
    
    test_texts = [
        "the",
        "language",
        "understanding",
        "pretraining",
        "natural language processing",
        "this is a test of the bpe tokenizer",
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        compression = len(text) / len(tokens) if len(tokens) > 0 else 0
        print(f"  '{text}'")
        print(f"    → {len(tokens)} tokens")
        print(f"    → 压缩率: {compression:.1f}x")
else:
    print(f"✗ 分词器文件不存在: {tokenizer_path}")

print("\n" + "="*60)
print("模型状态检查")
print("="*60)

if Path(model_path).exists():
    import torch
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"✓ 模型文件: {model_path}")
    print(f"  训练步数: {checkpoint['step']:,}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  最佳PPL: {checkpoint['best_ppl']:.1f}")
    print(f"  词汇表大小: {checkpoint['vocab_size']:,}")
else:
    print(f"✗ 模型文件不存在: {model_path}")
    print("  模型预训练尚未开始或未完成")
