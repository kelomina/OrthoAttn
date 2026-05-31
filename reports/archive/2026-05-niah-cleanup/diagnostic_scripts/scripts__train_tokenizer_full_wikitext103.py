"""使用完整WikiText-103训练集训练分词器（流式处理，低内存）。

直接从文件流式读取，避免一次性加载5.4亿字符到内存。
"""
from __future__ import annotations

import sys
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pretrain_hybrid_lm import FastBPETokenizer
from scripts.tiny_llama_shared import download_wikitext103

LOG_FILE = PROJECT_ROOT / "reports" / "tokenizer_training.log"


def log(msg, f=None):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if f:
        f.write(line + "\n")
        f.flush()


def main():
    with open(str(LOG_FILE), 'w', encoding='utf-8', buffering=1) as logf:
        log("="*70, logf)
        log("  使用完整WikiText-103训练分词器（流式处理）", logf)
        log("="*70, logf)
        
        # 1. 备份原分词器
        tokenizer_path = PROJECT_ROOT / "models" / "bpe_tokenizer.json"
        backup_path = PROJECT_ROOT / "models" / "bpe_tokenizer_backup_before_fulltrain.json"
        
        if tokenizer_path.exists():
            log("备份原分词器...", logf)
            shutil.copy2(str(tokenizer_path), str(backup_path))
            log(f"  备份至: {backup_path}", logf)
        
        # 2. 获取WikiText-103训练集文件路径
        data_dir = str(PROJECT_ROOT / "data" / "wikitext-103")
        log("获取WikiText-103数据集路径...", logf)
        splits = download_wikitext103(data_dir)
        train_file = splits['train']
        log(f"  训练集文件: {train_file}", logf)
        log(f"  文件大小: {train_file.stat().st_size / 1024 / 1024:.1f}MB", logf)
        
        # 3. 流式统计词频（直接从文件读取，不加载全部到内存）
        log("流式统计词频...", logf)
        word_freqs = defaultdict(int)
        total_chars = 0
        line_count = 0
        
        with open(str(train_file), 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                total_chars += len(line)
                for word in line.split():
                    word_freqs[word] += 1
                if line_count % 100000 == 0:
                    log(f"  已处理 {line_count:,} 行, {total_chars/1e6:.0f}M 字符, 唯一词: {len(word_freqs):,}", logf)
        
        log(f"词频统计完成: {total_chars:,} 字符, {len(word_freqs):,} 唯一词", logf)
        
        # 4. 创建tokenizer并训练
        log("创建tokenizer...", logf)
        tokenizer = FastBPETokenizer(vocab_size=32000)
        
        # 将词频字典转换为train方法需要的格式
        # 直接调用内部的训练逻辑，跳过text.split()步骤
        log("开始BPE训练...", logf)
        
        # 手动执行train的逻辑，使用已经统计好的词频
        # 首字符添加 ▁ 前缀标记 word 边界（SentencePiece 风格）
        word_list = []
        for word, count in word_freqs.items():
            chars = list(word)
            if chars:
                chars[0] = "▁" + chars[0]
            word_list.append((chars, count))
        del word_freqs
        
        # 初始化词汇表
        alphabet = set()
        for word, _ in word_list:
            alphabet.update(word)
        
        next_id = len(tokenizer.special_tokens)
        for char in sorted(alphabet):
            tokenizer.encoder[char] = next_id
            tokenizer.decoder[next_id] = char
            next_id += 1
        
        log(f"初始词汇表: {len(tokenizer.encoder)}", logf)
        
        # 增量式BPE合并
        num_merges = tokenizer.vocab_size - len(tokenizer.encoder)
        
        pair_freqs = defaultdict(int)
        pair_to_words = defaultdict(set)
        
        for word_idx, (word, freq) in enumerate(word_list):
            for j in range(len(word) - 1):
                pair = (word[j], word[j+1])
                pair_freqs[pair] += freq
                pair_to_words[pair].add(word_idx)
        
        log(f"初始pair数: {len(pair_freqs):,}", logf)
        
        for i in range(num_merges):
            if not pair_freqs:
                break
            
            best = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best]
            
            if best_freq < 2:
                break
            
            new_token = ''.join(best)
            new_id = len(tokenizer.encoder)
            
            tokenizer.encoder[new_token] = new_id
            tokenizer.decoder[new_id] = new_token
            tokenizer.merges.append(best)
            
            affected_word_indices = list(pair_to_words[best])
            
            for word_idx in affected_word_indices:
                word, freq = word_list[word_idx]
                
                for j in range(len(word) - 1):
                    old_pair = (word[j], word[j+1])
                    if old_pair in pair_freqs:
                        pair_freqs[old_pair] -= freq
                        if pair_freqs[old_pair] <= 0:
                            del pair_freqs[old_pair]
                        pair_to_words[old_pair].discard(word_idx)
                        if not pair_to_words[old_pair]:
                            del pair_to_words[old_pair]
                
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j+1]) == best:
                        new_word.append(new_token)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                
                word_list[word_idx] = (new_word, freq)
                
                for j in range(len(new_word) - 1):
                    new_pair = (new_word[j], new_word[j+1])
                    pair_freqs[new_pair] += freq
                    pair_to_words[new_pair].add(word_idx)
            
            if best in pair_freqs:
                del pair_freqs[best]
            if best in pair_to_words:
                del pair_to_words[best]
            
            if (i + 1) % 2000 == 0:
                log(f"  Merge {i+1}/{num_merges}: {best} (freq={best_freq:,})", logf)
        
        log(f"最终词汇表: {len(tokenizer.encoder):,}", logf)
        
        # 5. 测试编码和解码
        log("="*70, logf)
        log("  测试编码和解码", logf)
        log("="*70, logf)
        
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Natural language processing with Deep Learning.",
            "Machine Learning and Artificial Intelligence are related.",
            "Hello World! This is a test of proper casing.",
            "Deep learning models achieve state-of-the-art results.",
        ]
        
        for text in test_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            compression = len(text) / len(encoded) if len(encoded) > 0 else 0
            
            token_chunks = []
            for idx in encoded[:20]:
                if idx in tokenizer.decoder:
                    token_chunks.append(tokenizer.decoder[idx])
            
            log(f"原文: {text}", logf)
            log(f"  编码: {len(encoded)} tokens, 压缩率: {compression:.2f}x", logf)
            log(f"  Token: {'|'.join(token_chunks)}", logf)
            log(f"  解码: {decoded}", logf)
        
        # 6. 保存分词器
        log("="*70, logf)
        log("  保存分词器", logf)
        log("="*70, logf)
        
        full_path = PROJECT_ROOT / "models" / "bpe_tokenizer_full_wikitext103.json"
        tokenizer.save(str(full_path))
        log(f"保存至: {full_path}", logf)
        
        tokenizer.save(str(tokenizer_path))
        log(f"更新默认: {tokenizer_path}", logf)
        
        # 7. 验证大写支持
        log("="*70, logf)
        log("  验证大写支持", logf)
        log("="*70, logf)
        
        upper_chars = [c for c in tokenizer.encoder.keys() if len(c) == 1 and c.isupper()]
        lower_chars = [c for c in tokenizer.encoder.keys() if len(c) == 1 and c.islower()]
        log(f"大写字母: {len(upper_chars)} -> {upper_chars[:10]}", logf)
        log(f"小写字母: {len(lower_chars)} -> {lower_chars[:10]}", logf)
        
        log("="*70, logf)
        log("✅ 全量分词器训练完成！", logf)
        log(f"   训练数据: {total_chars:,} 字符", logf)
        log(f"   词汇表: {len(tokenizer.encoder):,} tokens", logf)
        log("="*70, logf)


if __name__ == "__main__":
    main()
