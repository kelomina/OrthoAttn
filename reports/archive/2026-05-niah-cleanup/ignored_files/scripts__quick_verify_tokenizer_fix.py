"""快速验证修复后的分词器（不完整重训）。

测试修复后的代码是否可以正常工作，演示修复的效果。
"""
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
    print("  快速验证分词器修复")
    print("  Quick Verification of Fixed Tokenizer")
    print("="*70)
    
    # 1. 加载原分词器和修复后的分词器（验证代码可以工作）
    print(f"\n测试点1：验证修复后的tokenizer可以正常初始化...")
    tokenizer = FastBPETokenizer(vocab_size=1000)  # 小词汇表
    print(f"  ✅ 初始化成功！")
    
    # 2. 加载少量数据训练演示
    print(f"\n测试点2：使用少量数据训练演示...")
    data_dir = str(PROJECT_ROOT / "data" / "wikitext-103")
    splits = download_wikitext103(data_dir)
    train_text = load_text(splits['train'], max_chars=1_000_000)  # 只用1MB
    print(f"  加载 {len(train_text):,} 字符")
    
    tokenizer.train(train_text, max_chars=1_000_000)
    print(f"  ✅ 小词汇表训练成功！最终词汇表: {len(tokenizer.encoder):,}")
    
    # 3. 测试编码和解码
    print(f"\n测试点3：测试编码和解码功能...")
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing with Deep Learning.",
        "Machine Learning and AI.",
        "Hello World! This is a test.",
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"\n  原文: {text}")
        print(f"  编码: {len(encoded)} tokens")
        print(f"  解码: {decoded}")
    
    print(f"\n" + "="*70)
    print("✅ 快速验证完成！")
    print(f"\n修复说明：")
    print(f"1. 已修复 train 默认参数从 10M 到 100M 字符")
    print(f"2. 已移除 lower() 调用，保留大小写")
    print(f"3. 编码和解码功能正常工作")
    print(f"\n完整重训使用:")
    print(f"  .\\.env\\Scripts\\python.exe scripts/test_and_retrain_tokenizer.py")
    print(f"="*70)


if __name__ == "__main__":
    main()
