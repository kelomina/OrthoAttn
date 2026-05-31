"""测试修复后的FastBPETokenizer。

对比原分词器和修复后的分词器效果。
"""
from __future__ import annotations

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pretrain_hybrid_lm import FastBPETokenizer
from scripts.tiny_llama_shared import download_wikitext103, load_text


def main():
    print("="*70)
    print("  测试修复后的FastBPETokenizer")
    print("  Test Fixed FastBPETokenizer")
    print("="*70)
    
    # 1. 加载和备份原分词器
    tokenizer_path = PROJECT_ROOT / "models" / "bpe_tokenizer.json"
    backup_path = PROJECT_ROOT / "models" / "bpe_tokenizer_backup_before_fix.json"
    
    if tokenizer_path.exists():
        print(f"\n备份原分词器...")
        shutil.copy2(str(tokenizer_path), str(backup_path))
        print(f"  备份至: {backup_path}")
        
        print(f"\n加载原分词器...")
        original_tokenizer = FastBPETokenizer.load(str(backup_path))
        print(f"  原词汇表大小: {len(original_tokenizer.encoder):,}")
    
    # 2. 下载和加载训练数据
    data_dir = str(PROJECT_ROOT / "data" / "wikitext-103")
    print(f"\n加载WikiText-103...")
    splits = download_wikitext103(data_dir)
    
    print(f"加载训练集...")
    train_text = load_text(splits['train'], max_chars=200_000_000)  # 先用2亿字符
    print(f"训练集大小: {len(train_text):,} 字符")
    
    # 3. 测试修复后的分词器
    print(f"\n创建和训练修复后的分词器...")
    tokenizer = FastBPETokenizer(vocab_size=32000)
    tokenizer.train(train_text, max_chars=200_000_000)
    
    # 4. 测试编码和解码
    print(f"\n" + "="*70)
    print(f"  测试编码和解码")
    print(f"="*70)
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing with Deep Learning.",
        "Machine Learning and Artificial Intelligence.",
        "Wikitext-103 is a large language modeling dataset.",
        "Hello World! This is a test of proper casing.",
        "This sentence uses correct punctuation.",
    ]
    
    for text in test_texts:
        print(f"\n原文: {text}")
        encoded = tokenizer.encode(text)
        print(f"编码: {len(encoded)} tokens")
        
        # 可视化前20个token对应的文本
        token_chunks = []
        for idx in encoded[:20]:
            if idx in tokenizer.decoder:
                token_chunks.append(tokenizer.decoder[idx])
        print(f"Token chunks (前20): {'|'.join(token_chunks)}")
        
        decoded = tokenizer.decode(encoded)
        print(f"解码: {decoded}")
        
        compression = len(text) / len(encoded) if len(encoded) > 0 else 0
        print(f"压缩率: {compression:.2f}x")
    
    # 5. 保存修复后的分词器
    print(f"\n" + "="*70)
    print(f"  保存修复后的分词器")
    print(f"="*70)
    new_tokenizer_path = PROJECT_ROOT / "models" / "bpe_tokenizer_fixed.json"
    tokenizer.save(str(new_tokenizer_path))
    print(f"保存至: {new_tokenizer_path}")
    
    # 也覆盖默认的
    tokenizer.save(str(tokenizer_path))
    print(f"同时更新默认分词器: {tokenizer_path}")
    
    print(f"\n" + "="*70)
    print("✅ 测试和重训完成！")
    print("="*70)


if __name__ == "__main__":
    main()
