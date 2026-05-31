"""验证现有的分词器和WikiText-103数据集。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pretrain_hybrid_lm import FastBPETokenizer
from scripts.tiny_llama_shared import download_wikitext103, load_text


def main():
    print("="*70)
    print("  验证现有分词器和WikiText-103数据集")
    print("  Verify Existing Tokenizer and WikiText-103 Dataset")
    print("="*70)
    
    # 1. 检查WikiText-103数据
    print("\n检查WikiText-103数据集...")
    data_dir = str(PROJECT_ROOT / "data" / "wikitext-103")
    splits = download_wikitext103(data_dir)
    
    for split_name, path in splits.items():
        print(f"\n{split_name}:")
        print(f"  路径: {path}")
        print(f"  存在: {'✅' if path.exists() else '❌'}")
        if path.exists():
            text = load_text(path)
            print(f"  大小: {len(text):,} 字符")
            print(f"  样本: {text[:100]}...")
    
    # 2. 检查分词器
    tokenizer_path = PROJECT_ROOT / "models" / "bpe_tokenizer.json"
    print("\n" + "="*70)
    print(f"检查分词器: {tokenizer_path}")
    print("="*70)
    
    if tokenizer_path.exists():
        tokenizer = FastBPETokenizer.load(str(tokenizer_path))
        print(f"  ✅ 加载成功")
        print(f"  词汇表大小: {len(tokenizer.encoder):,}")
        print(f"  合并规则数: {len(tokenizer.merges):,}")
        
        # 显示一些样本
        print(f"\n  特殊tokens: {tokenizer.special_tokens}")
        print(f"\n  前30个单字符tokens:")
        for i, (token, idx) in enumerate(tokenizer.encoder.items()):
            if i >= 30:
                break
            if len(token) == 1:  # 只显示单字符
                print(f"    {token} -> {idx}")
        
        # 显示一些长token
        print(f"\n  一些长BPE tokens:")
        long_tokens = [t for t in tokenizer.encoder.keys() if len(t) > 5]
        for i, token in enumerate(long_tokens[:10]):
            print(f"    {token} -> {tokenizer.encoder[token]}")
        
        # 测试编码和解码
        print(f"\n测试编码和解码...")
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Natural language processing with deep learning.",
            "WikiText-103 is a large-scale language modeling dataset.",
        ]
        for text in test_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            print(f"\n  原文: {text}")
            print(f"  编码: {len(encoded)} tokens -> {encoded[:10]}...")
            print(f"  解码: {decoded}")
            compression = len(text) / len(encoded) if len(encoded) > 0 else 0
            print(f"  压缩率: {compression:.2f}x")
    else:
        print(f"  ❌ 分词器不存在: {tokenizer_path}")
    
    print("\n" + "="*70)
    print("✅ 验证完成！")
    print("="*70)


if __name__ == "__main__":
    main()
