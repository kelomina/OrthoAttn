"""使用完整WikiText-103训练BPE分词器（简化版）。

这个脚本直接复用原有的FastBPETokenizer，但使用完整训练集。
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
    print("  使用完整WikiText-103训练BPE分词器")
    print("  Train BPE Tokenizer on Full WikiText-103")
    print("="*70)
    
    # 1. 下载WikiText-103
    data_dir = str(PROJECT_ROOT / "data" / "wikitext-103")
    print(f"\n加载WikiText-103...")
    splits = download_wikitext103(data_dir)
    print(f"数据集路径: {splits['train']}")
    
    # 2. 加载完整训练集
    print("\n加载训练集...")
    train_text = load_text(splits['train'], max_chars=None)  # 使用全部数据
    print(f"训练集大小: {len(train_text):,} 字符")
    
    # 3. 检查现有的分词器是否已经使用完整训练集
    tokenizer_path = PROJECT_ROOT / "models" / "bpe_tokenizer.json"
    if tokenizer_path.exists():
        print(f"\n现有分词器已存在于: {tokenizer_path}")
        print("是否要重新训练？")
        # 直接加载现有分词器作为参考
        existing_tokenizer = FastBPETokenizer.load(str(tokenizer_path))
        print(f"现有词汇表大小: {len(existing_tokenizer.encoder):,}")
    
    # 4. 训练新分词器（使用完整训练集，增加max_chars）
    print("\n开始训练分词器...")
    tokenizer = FastBPETokenizer(vocab_size=32000)
    
    # 使用更多训练数据（500M字符而不是原来的10M）
    tokenizer.train(train_text, max_chars=500_000_000)  # 5亿字符
    
    # 5. 保存新分词器
    save_path = PROJECT_ROOT / "models" / "bpe_tokenizer_full_wikitext103.json"
    tokenizer.save(str(save_path))
    print(f"\n分词器已保存至: {save_path}")
    
    # 6. 验证和测试
    print("\n测试分词器...")
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is fun!",
        "Machine learning is a subset of artificial intelligence.",
        "WikiText-103 is a large language modeling dataset.",
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"\n原文: {text}")
        print(f"Token: {len(encoded)} tokens")
        print(f"解码: {decoded}")
        
        compression = len(text) / len(encoded) if len(encoded) > 0 else 0
        print(f"压缩率: {compression:.2f}x")
    
    # 7. 也覆盖现有的分词器（可选，先备份）
    backup_path = PROJECT_ROOT / "models" / "bpe_tokenizer_original_backup.json"
    if tokenizer_path.exists() and not backup_path.exists():
        import shutil
        shutil.copy2(str(tokenizer_path), str(backup_path))
        print(f"\n原分词器已备份至: {backup_path}")
    
    tokenizer.save(str(tokenizer_path))
    print(f"\n也已保存为默认分词器: {tokenizer_path}")
    
    print("\n" + "="*70)
    print("✅ 分词器训练完成！")
    print("="*70)


if __name__ == "__main__":
    main()
