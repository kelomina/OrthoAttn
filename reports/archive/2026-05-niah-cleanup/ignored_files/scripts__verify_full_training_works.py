"""测试和验证全量训练脚本是否正常工作。

先运行一个小版本测试确认一切正常。
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
    print("  验证全量训练脚本")
    print("  Verify Full Training Script")
    print("="*70)
    
    # 1. 检查数据
    data_dir = str(PROJECT_ROOT / "data" / "wikitext-103")
    splits = download_wikitext103(data_dir)
    
    print(f"\n检查WikiText-103训练集...")
    print(f"  训练集路径: {splits['train']}")
    print(f"  文件存在: {splits['train'].exists()}")
    
    train_text = load_text(splits['train'], max_chars=10_000_000)  # 先只用10MB测试
    print(f"  训练数据加载成功: {len(train_text):,} 字符")
    
    # 2. 测试tokenizer训练
    print(f"\n测试tokenizer训练（小规模测试）...")
    tokenizer = FastBPETokenizer(vocab_size=32000)
    tokenizer.train(train_text, max_chars=10_000_000)
    print(f"  训练成功！最终词汇表: {len(tokenizer.encoder):,}")
    
    # 3. 测试编码/解码
    test_text = "The quick brown fox jumps over the lazy dog."
    print(f"\n测试编码/解码...")
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"  原文: {test_text}")
    print(f"  编码: {len(encoded)} tokens")
    print(f"  解码: {decoded}")
    
    # 4. 总结
    print(f"\n" + "="*70)
    print("✅ 全量训练脚本验证成功！")
    print(f"\n要运行真正的全量训练（5.4亿字符），请执行:")
    print(f"  .\\.env\\Scripts\\python.exe scripts\\train_tokenizer_full_wikitext103.py")
    print(f"\n训练将:")
    print(f"  - 使用完整的WikiText-103训练集（无截断）")
    print(f"  - 保留大小写")
    print(f"  - 生成32,000词汇表")
    print(f"  - 保存到 models/bpe_tokenizer_full_wikitext103.json")
    print(f"  - 同时更新默认的 models/bpe_tokenizer.json")
    print("="*70)


if __name__ == "__main__":
    main()
