"""使用完整WikiText-103训练改进版的FastBPETokenizer。

这个脚本包含：
1. 对原分词器的改进（大小写保留、更好的BPE逻辑）
2. 使用完整WikiText-103训练
3. 验证和测试工具
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pretrain_hybrid_lm import FastBPETokenizer as OriginalFastBPETokenizer
from scripts.tiny_llama_shared import download_wikitext103, load_text


class ImprovedFastBPETokenizer(OriginalFastBPETokenizer):
    """改进版的FastBPETokenizer。
    
    改进点：
    1. 保留大小写（原来全部lower()）
    2. 更好的合并逻辑
    3. 支持更多的训练选项
    """
    
    def train(self, text, max_chars=None, min_freq=2):
        """改进版的BPE训练。
        
        Args:
            text: 训练文本
            max_chars: 最大使用字符数（None表示使用全部）
            min_freq: 最小词频
        """
        if max_chars:
            text = text[:max_chars]
        print(f"训练数据: {len(text):,} 字符")
        
        # 1. 统计词频（保留大小写）
        words = text.split()  # 不使用lower()
        word_freqs = Counter(words)
        
        # 过滤低频词
        if min_freq > 1:
            word_freqs = {w: cnt for w, cnt in word_freqs.items() if cnt >= min_freq}
            print(f"  过滤后词汇: {len(word_freqs):,} (min_freq={min_freq})")
        
        # 2. 转换为字符元组
        word_freqs = {tuple(word): count for word, count in word_freqs.items()}
        
        # 3. 初始化词汇表（所有字符）
        alphabet = set()
        for word in word_freqs:
            alphabet.update(word)
        
        next_id = len(self.special_tokens)
        for char in sorted(alphabet):
            self.encoder[char] = next_id
            self.decoder[next_id] = char
            next_id += 1
        
        print(f"初始词汇表: {len(self.encoder)}")
        
        # 4. BPE合并
        num_merges = self.vocab_size - len(self.encoder)
        print(f"需要执行 {num_merges:,} 次合并")
        
        for i in range(num_merges):
            # 统计pair频率
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j+1])] += freq
            
            if not pairs:
                break
            
            # 选择最高频pair
            best = max(pairs, key=pairs.get)
            new_token = ''.join(best)
            new_id = len(self.encoder)
            
            self.encoder[new_token] = new_id
            self.decoder[new_id] = new_token
            self.merges.append(best)
            
            # 合并词汇
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j+1]) == best:
                        new_word.append(new_token)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs
            
            if (i + 1) % 5000 == 0:
                print(f"  Merge {i+1:,}/{num_merges:,}: {best} (freq={pairs[best]:,})")
        
        print(f"最终词汇表: {len(self.encoder):,}")
    
    def encode(self, text, add_special_tokens=True):
        """编码文本为token IDs（保留大小写）。"""
        words = text.split()  # 不使用lower()
        tokens = []
        for word in words:
            word = tuple(word)
            while len(word) > 0:
                matched = False
                for length in range(min(len(word), 20), 0, -1):
                    subword = ''.join(word[:length])
                    if subword in self.encoder:
                        tokens.append(self.encoder[subword])
                        word = word[length:]
                        matched = True
                        break
                if not matched:
                    tokens.append(self.special_tokens["<unk>"])
                    word = word[1:]

        if add_special_tokens:
            return [self.special_tokens["<s>"]] + tokens + [self.special_tokens["</s>"]]
        return tokens
    
    def decode(self, ids):
        """解码token IDs为文本（保留token之间的空格）。"""
        tokens = []
        for id in ids:
            if id in self.decoder:
                tok = self.decoder[id]
                if tok not in self.special_tokens:
                    tokens.append(tok)
        return ' '.join(tokens)


def main():
    print("="*70)
    print("  使用完整WikiText-103训练改进版分词器")
    print("  Train Improved BPE Tokenizer on Full WikiText-103")
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
    
    # 3. 创建改进版分词器
    print("\n初始化分词器...")
    tokenizer = ImprovedFastBPETokenizer(vocab_size=32000)
    
    # 4. 训练分词器
    print("\n开始训练分词器...")
    print(f"  词汇表大小: {tokenizer.vocab_size:,}")
    tokenizer.train(train_text, max_chars=None, min_freq=2)
    
    # 5. 保存改进版分词器
    save_path_improved = PROJECT_ROOT / "models" / "bpe_tokenizer_improved.json"
    tokenizer.save(str(save_path_improved))
    
    # 6. 验证和测试
    print("\n测试分词器...")
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is fun!",
        "Machine learning is a subset of artificial intelligence.",
        "WikiText-103 is a large language modeling dataset.",
        "Hello World! This is a test.",
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"\n原文: {text}")
        print(f"Token: {len(encoded)} tokens")
        print(f"解码: {decoded}")
        
        # 计算压缩率
        compression = len(text) / len(encoded) if len(encoded) > 0 else 0
        print(f"压缩率: {compression:.2f}x")
    
    # 7. 也为了兼容性保存原始格式的分词器（可选）
    print("\n也保存一份格式兼容的原始分词器...")
    original_tokenizer = OriginalFastBPETokenizer(vocab_size=32000)
    original_tokenizer.encoder = tokenizer.encoder
    original_tokenizer.decoder = tokenizer.decoder
    original_tokenizer.merges = tokenizer.merges
    original_tokenizer.save(str(PROJECT_ROOT / "models" / "bpe_tokenizer_full_wikitext103.json"))
    
    print("\n" + "="*70)
    print("✅ 分词器训练完成！")
    print("="*70)


if __name__ == "__main__":
    main()
