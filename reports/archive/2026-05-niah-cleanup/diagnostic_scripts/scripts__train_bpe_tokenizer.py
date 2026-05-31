"""训练 BPE 分词器用于语言模型预训练。

这个脚本从 WikiText-103 数据集训练一个高效的 BPE 分词器，
最大化减少 token 数量，提高训练效率。

Training a BPE tokenizer for language model pre-training.
This script trains an efficient BPE tokenizer from WikiText-103 dataset
to minimize token count and improve training efficiency.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

# 确保可以导入项目
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny_llama_shared import download_wikitext103, load_text


class BPETokenizer:
    """Byte-Pair Encoding 分词器实现。
    
    Byte-Pair Encoding tokenizer implementation.
    BPE 通过迭代合并最频繁出现的字符对来构建词汇表，
    在字符级和词级之间找到平衡，显著减少 token 数量。
    
    BPE builds vocabulary by iteratively merging the most frequent character pairs,
    finding a balance between character-level and word-level tokenization,
    significantly reducing token count.
    """
    
    def __init__(self, vocab_size: int = 30000, max_token_length: int = 64):
        """初始化 BPE 分词器。
        
        Args:
            vocab_size: 目标词汇表大小（默认 30000）
            max_token_length: 最大 token 长度（默认 64）
        """
        self.vocab_size = vocab_size
        self.max_token_length = max_token_length
        self.vocab = {}  # token -> id
        self.merges = []  # list of (token1, token2) -> new_token
        self.encoder = {}  # token -> id
        self.decoder = {}  # id -> token
        
        # 特殊 tokens
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        
        self.special_tokens = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
        }
    
    def _get_word_freqs(self, text: str) -> dict[tuple[str, ...], int]:
        """统计词频。
        
        Count word frequencies.
        将文本分割为单词，并统计每个单词的出现频率。
        
        Args:
            text: 输入文本
            
        Returns:
            词频字典 {word_tuple: count}
        """
        # 简单的单词分割
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        word_freqs = Counter(words)
        
        # 转换为字符元组
        word_freq_tuples = {
            tuple(word): count for word, count in word_freqs.items()
        }
        
        return word_freq_tuples
    
    def _get_stats(self, word_freqs: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
        """统计字符对频率。
        
        Count character pair frequencies.
        
        Args:
            word_freqs: 词频字典
            
        Returns:
            字符对频率字典 {(char1, char2): count}
        """
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs
    
    def _merge_vocab(self, pair: tuple[str, str], word_freqs: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
        """合并词汇表中的字符对。
        
        Merge a character pair in the vocabulary.
        
        Args:
            pair: 要合并的字符对
            word_freqs: 词频字典
            
        Returns:
            更新后的词频字典
        """
        new_word_freqs = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in word_freqs:
            new_word = pattern.sub(''.join(pair), ' '.join(word))
            new_word = tuple(new_word.split())
            new_word_freqs[new_word] = word_freqs[word]
        
        return new_word_freqs
    
    def train(self, text: str):
        """训练 BPE 分词器。
        
        Train the BPE tokenizer.
        
        Args:
            text: 训练文本
        """
        print(f"开始训练 BPE 分词器，目标词汇表大小: {self.vocab_size}")
        print(f"训练文本长度: {len(text):,} 字符")
        
        # 初始化：字符级词汇表
        word_freqs = self._get_word_freqs(text)
        
        # 获取所有初始字符
        alphabet = set()
        for word in word_freqs.keys():
            for char in word:
                alphabet.add(char)
        
        # 初始化词汇表
        self.encoder = {char: idx + len(self.special_tokens) for idx, char in enumerate(sorted(alphabet))}
        self.decoder = {idx: char for char, idx in self.encoder.items()}
        
        # 添加特殊 tokens
        self.encoder.update({token: idx for token, idx in self.special_tokens.items()})
        self.decoder.update({idx: token for token, idx in self.special_tokens.items()})
        
        print(f"初始词汇表大小: {len(self.encoder)}")
        
        # 迭代合并
        num_merges = self.vocab_size - len(self.encoder)
        for i in range(num_merges):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            
            # 找到最频繁的字符对
            best_pair = max(pairs, key=pairs.get)
            bigram_freq = pairs[best_pair]
            
            # 合并
            new_token = ''.join(best_pair)
            new_id = len(self.encoder)
            
            self.encoder[new_token] = new_id
            self.decoder[new_id] = new_token
            self.merges.append(best_pair)
            
            word_freqs = self._merge_vocab(best_pair, word_freqs)
            
            if (i + 1) % 1000 == 0:
                print(f"  合并 {i+1}/{num_merges}: {best_pair} -> '{new_token}' (频率: {bigram_freq})")
                print(f"  当前词汇表大小: {len(self.encoder)}")
        
        self.vocab = self.encoder
        print(f"训练完成！最终词汇表大小: {len(self.vocab)}")
    
    def encode(self, text: str) -> list[int]:
        """将文本编码为 token IDs。
        
        Encode text to token IDs.
        
        Args:
            text: 输入文本
            
        Returns:
            token ID 列表
        """
        # 首先按字符分割
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        tokens = []
        
        for word in words:
            # 尝试匹配最长子词
            remaining = word
            while remaining:
                matched = False
                # 从最长到最短尝试匹配
                for length in range(min(len(remaining), self.max_token_length), 0, -1):
                    subword = remaining[:length]
                    if subword in self.encoder:
                        tokens.append(self.encoder[subword])
                        remaining = remaining[length:]
                        matched = True
                        break
                if not matched:
                    # 如果找不到，使用 UNK
                    tokens.append(self.special_tokens[self.unk_token])
                    remaining = remaining[1:]
        
        return tokens
    
    def decode(self, ids: list[int]) -> str:
        """将 token IDs 解码为文本。
        
        Decode token IDs to text.
        
        Args:
            ids: token ID 列表
            
        Returns:
            解码后的文本
        """
        tokens = []
        for id in ids:
            if id in self.decoder:
                token = self.decoder[id]
                if token not in self.special_tokens:
                    tokens.append(token)
        return ''.join(tokens)
    
    def save(self, path: str):
        """保存分词器到文件。
        
        Save tokenizer to file.
        
        Args:
            path: 保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        tokenizer_data = {
            "vocab_size": self.vocab_size,
            "max_token_length": self.max_token_length,
            "vocab": self.vocab,
            "merges": [[a, b] for a, b in self.merges],
            "special_tokens": self.special_tokens,
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        
        print(f"分词器已保存到: {path}")
    
    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """从文件加载分词器。
        
        Load tokenizer from file.
        
        Args:
            path: 文件路径
            
        Returns:
            BPETokenizer 实例
        """
        with open(path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        
        tokenizer = cls(
            vocab_size=tokenizer_data["vocab_size"],
            max_token_length=tokenizer_data.get("max_token_length", 64),
        )
        tokenizer.vocab = tokenizer_data["vocab"]
        tokenizer.encoder = tokenizer_data["vocab"]
        tokenizer.decoder = {v: k for k, v in tokenizer_data["vocab"].items()}
        tokenizer.merges = [tuple(m) for m in tokenizer_data["merges"]]
        tokenizer.special_tokens = tokenizer_data["special_tokens"]
        
        return tokenizer


def train_tokenizer_from_wikitext(
    vocab_size: int = 30000,
    data_dir: str = "data/wikitext-103",
    output_path: str = "models/bpe_tokenizer.json",
    max_train_chars: int = 50_000_000,  # 50M 字符用于训练
):
    """从 WikiText-103 训练 BPE 分词器。
    
    Train BPE tokenizer from WikiText-103.
    
    Args:
        vocab_size: 目标词汇表大小
        data_dir: 数据目录
        output_path: 输出路径
        max_train_chars: 最大训练字符数
    """
    print("="*60)
    print("训练 BPE 分词器")
    print("="*60)
    
    # 下载数据
    data_paths = download_wikitext103(data_dir)
    train_text = load_text(data_paths["train"], max_chars=max_train_chars)
    
    print(f"训练文本: {len(train_text):,} 字符")
    print(f"目标词汇表大小: {vocab_size:,}")
    print()
    
    # 训练分词器
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(train_text)
    
    # 保存
    tokenizer.save(output_path)
    
    # 测试
    print("\n" + "="*60)
    print("测试分词器")
    print("="*60)
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Language models are becoming increasingly powerful.",
        "This is a test of the BPE tokenizer.",
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"原文: {text}")
        print(f"编码: {len(tokens)} tokens")
        print(f"解码: {decoded}")
        print(f"压缩率: {len(text) / len(tokens):.1f}x")
        print()
    
    return tokenizer


if __name__ == "__main__":
    train_tokenizer_from_wikitext(
        vocab_size=30000,
        max_train_chars=50_000_000,
    )
