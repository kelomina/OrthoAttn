"""使用真实数据预训练混合架构语言模型。

包含：
1. 高效BPE分词器训练
2. WikiText-103数据加载
3. 混合架构模型预训练
4. 训练监控和保存

Pre-training hybrid architecture language model with real data.
Includes:
1. Efficient BPE tokenizer training
2. WikiText-103 data loading  
3. Hybrid architecture model pre-training
4. Training monitoring and saving
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny_llama_shared import download_wikitext103, load_text
from src.dsra.mhdsra2.improved_dsra_mha import MultiHeadDSRA2, MHDSRA2Config
from src.dsra.swanlab_utils import init_swanlab

SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FastBPETokenizer:
    """高效的BPE分词器。
    
    Efficient BPE tokenizer.
    使用简化算法快速训练，最大化token压缩率。
    """
    
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.encoder = {}  # str -> int
        self.decoder = {}  # int -> str
        self.merges = []
        
        # 特殊token
        self.special_tokens = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        self.encoder.update(self.special_tokens)
        self.decoder.update({v: k for k, v in self.special_tokens.items()})
    
    def train(self, text, max_chars=100_000_000):
        """训练分词器（增量式BPE，支持全量数据高效训练）。
        
        Args:
            text: 训练文本
            max_chars: 最大使用字符数（None表示使用全部）
        """
        if max_chars is not None:
            text = text[:max_chars]
        print(f"训练数据: {len(text):,} 字符", flush=True)
        
        # 1. 流式统计词频（避免一次性split创建巨大列表）
        word_freqs_raw = defaultdict(int)
        start = 0
        text_len = len(text)
        chunk_size = 10_000_000
        counted = 0
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            if start > 0:
                # 确保不在词中间切分：找到第一个空格
                space_pos = chunk.find(' ')
                if space_pos >= 0:
                    chunk = chunk[space_pos + 1:]
                else:
                    start = end
                    continue
            for word in chunk.split():
                word_freqs_raw[word] += 1
            counted += 1
            if counted % 10 == 0:
                print(f"  词频统计: {end/text_len*100:.0f}%, 唯一词: {len(word_freqs_raw):,}", flush=True)
            start = end
        
        print(f"唯一词数: {len(word_freqs_raw):,}", flush=True)
        
        # 2. 转换为字符列表，同时记录每个词的频率
        # 首字符添加 ▁ 前缀标记 word 边界（SentencePiece 风格）
        word_list = []
        for word, count in word_freqs_raw.items():
            chars = list(word)
            if chars:
                chars[0] = "▁" + chars[0]
            word_list.append((chars, count))
        del word_freqs_raw
        
        # 3. 初始化词汇表（所有字符，含 ▁ 前缀版本）
        alphabet = set()
        for word, _ in word_list:
            alphabet.update(word)
        
        next_id = len(self.special_tokens)
        for char in sorted(alphabet):
            self.encoder[char] = next_id
            self.decoder[next_id] = char
            next_id += 1
        
        print(f"初始词汇表: {len(self.encoder)}", flush=True)
        
        # 4. 增量式BPE合并
        num_merges = self.vocab_size - len(self.encoder)
        
        # 预计算所有pair频率
        pair_freqs = defaultdict(int)
        pair_to_words = defaultdict(set)
        
        for word_idx, (word, freq) in enumerate(word_list):
            for j in range(len(word) - 1):
                pair = (word[j], word[j+1])
                pair_freqs[pair] += freq
                pair_to_words[pair].add(word_idx)
        
        print(f"初始pair数: {len(pair_freqs):,}", flush=True)
        
        for i in range(num_merges):
            if not pair_freqs:
                break
            
            best = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best]
            
            if best_freq < 2:
                break
            
            new_token = ''.join(best)
            new_id = len(self.encoder)
            
            self.encoder[new_token] = new_id
            self.decoder[new_id] = new_token
            self.merges.append(best)
            
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
                print(f"  Merge {i+1}/{num_merges}: {best} (freq={best_freq:,})", flush=True)
        
        print(f"最终词汇表: {len(self.encoder):,}", flush=True)
    
    def encode(self, text, add_special_tokens=True):
        """编码文本为token IDs。

        中文说明:
        - 调用方 / Called by: StreamingDataset.__init__, scripts/pretrain_hybrid_lm
        - 作用 / Purpose: 将文本转换为 BPE token ID 序列
        - 参数 / Parameters:
          text: 输入文本
          add_special_tokens: 是否在序列首尾添加 <s> 和 </s> token，默认 True
            当分块编码时应设为 False，由调用方统一添加边界 token
        - 返回值 / Returns: token ID 列表
        - 关键词 / Keywords: BPE, encode, tokenizer, special_tokens, 编码
        - 注意 / Note: 每个 word 的首子 token 带 ▁ 前缀标记 word 边界，
          BPE 匹配时首字符使用 ▁ 前缀版本进行贪婪最长匹配
        """
        # 标点符号预处理：将标点符号与单词分开（连字符在单词内部时保留）
        import re
        # 定义标点符号（排除连字符，因为它常用在单词内部）
        punctuation = r'[^\w\s\-]'
        # 在标点符号前后添加空格
        processed_text = re.sub(punctuation, r' \g<0> ', text)
        # 去除多余空格
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        words = processed_text.split()
        tokens = []
        for word in words:
            word = tuple(word)
            first = True
            while len(word) > 0:
                matched = False
                for length in range(min(len(word), 20), 0, -1):
                    subword = ''.join(word[:length])
                    if first:
                        sp_key = "▁" + subword
                        if sp_key in self.encoder:
                            tokens.append(self.encoder[sp_key])
                            word = word[length:]
                            matched = True
                            first = False
                            break
                    if subword in self.encoder:
                        tokens.append(self.encoder[subword])
                        word = word[length:]
                        if first:
                            first = False
                        matched = True
                        break
                if not matched:
                    tokens.append(self.special_tokens["<unk>"])
                    word = word[1:]
                    if first:
                        first = False

        if add_special_tokens:
            return [self.special_tokens["<s>"]] + tokens + [self.special_tokens["</s>"]]
        return tokens
    
    def decode(self, ids):
        """解码token IDs为文本。

        中文说明:
        - 调用方 / Called by: chat.py, 验证脚本
        - 作用 / Purpose: 将 token ID 序列还原为文本
        - 参数 / Parameters: ids - token ID 列表
        - 返回值 / Returns: 还原后的文本字符串
        - 关键词 / Keywords: BPE, decode, tokenizer, 解码
        - 注意 / Note: ▁ 前缀表示 word 边界，替换为空格；无 ▁ 的 token 与前一个 token 连接；
          标点符号前的空格会被去除
        """
        import re
        parts = []
        for id in ids:
            if id in self.decoder:
                tok = self.decoder[id]
                if tok in self.special_tokens:
                    continue
                if tok.startswith("▁"):
                    parts.append(" " + tok[1:])
                else:
                    parts.append(tok)
        # 拼接后处理标点符号前的空格
        decoded = ''.join(parts).strip()
        # 去除标点符号前的空格
        decoded = re.sub(r' ([^\w\s])', r'\1', decoded)
        return decoded
    
    def save(self, path):
        """保存分词器"""
        data = {
            "vocab_size": self.vocab_size,
            "encoder": self.encoder,
            "merges": [list(m) for m in self.merges],
            "special_tokens": self.special_tokens,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"分词器已保存: {path}")
    
    @classmethod
    def load(cls, path):
        """加载分词器"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.encoder = data["encoder"]
        tokenizer.decoder = {v: k for k, v in data["encoder"].items()}
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.special_tokens = data["special_tokens"]
        return tokenizer


class StreamingDataset(Dataset):
    """流式数据集，支持分块加载。
    
    Streaming dataset with chunked loading support.
    内存高效实现，按需加载数据块。
    """
    
    def __init__(self, tokenizer, text, seq_len=512):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # 编码整个文本（分块编码，避免一次性占用过多内存）
        # encode chunks without per-chunk BOS/EOS to avoid data corruption;
        # add <s> and </s> only at the full-text boundaries below
        print(f"编码文本: {len(text):,} 字符")
        chunk_size = 1_000_000
        all_tokens = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            tokens = tokenizer.encode(chunk, add_special_tokens=False)
            all_tokens.extend(tokens)
        # 仅在完整文本首尾添加 BOS/EOS
        all_tokens = [tokenizer.special_tokens["<s>"]] + all_tokens + [tokenizer.special_tokens["</s>"]]
        
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.num_samples = (len(self.tokens) - 1) // seq_len
        
        print(f"总tokens: {len(self.tokens):,}")
        print(f"样本数: {self.num_samples:,}")
        print(f"压缩率: {len(text) / len(self.tokens):.2f}x")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        return self.tokens[start:end-1], self.tokens[start+1:end]


class HybridLanguageModel(nn.Module):
    """混合架构语言模型 (ST + MHDSRA2)。
    
    Hybrid architecture language model (ST + MHDSRA2).
    ST 和 MHDSRA2 并行处理，每层输出级融合。
    每层 MHDSRA2 维护独立状态，避免层间状态覆盖。
    """
    
    def __init__(self, vocab_size, dim=256, n_layers=4, n_heads=4, 
                 slots=128, local_window=512, chunk_size=512,
                 use_gradient_checkpointing=False):
        super().__init__()
        self.chunk_size = chunk_size
        self.dim = dim
        self.n_layers = n_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self._causal_mask_cache = {}
        
        # Embedding
        self.tok_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(chunk_size, dim)
        
        # MHDSRA2分支 - 预训练场景优化配置
        # read_topk/write_topk: 增大以提升slot信息流通量，让MH分支能访问更多上下文
        # tau_write_init: 提高以锐化写入路由，减少slot混叠
        # eta: 提高以加快写入门饱和，加速slot稳定
        mh_cfg = MHDSRA2Config(
            dim=dim, heads=n_heads,
            local_window=local_window,
            slot_pe="rope", slots=slots,
            tau_init=8.0, tau_write_init=8.0,
            read_topk=8, write_topk=4,
            use_retrieval=False,
            forget_base=0.001,
            usage_decay=0.995,
            conf_decay=0.999,
            eta=2.0,
            max_update=1.0,
            hard_read=True,
        )
        self.mh_layers = nn.ModuleList([
            MultiHeadDSRA2(mh_cfg) for _ in range(n_layers)
        ])
        
        # MHDSRA2 输出层归一化和缩放 - 平衡两个分支的输出幅度
        self.mh_out_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(n_layers)
        ])
        # ST 输出层归一化 - 与 MH 分支对称，消除幅度差异
        self.st_out_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(n_layers)
        ])
        # 初始值 1.0：中性起点，训练中自适应调整
        self.mh_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(n_layers)
        ])
        
        # ST分支 - 每层一个轻量ST，与MHDSRA2层对齐
        self.st_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=dim*4,
                batch_first=True, activation='gelu'
            ) for _ in range(n_layers)
        ])
        self.st_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(n_layers)
        ])
        
        # 每层独立融合门控 - 基于两个分支输出动态决定融合权重
        self.fuse_gates = nn.ModuleList([
            nn.Linear(dim * 2, 2) for _ in range(n_layers)
        ])
        # 门控初始化修复：bias=0，weight小值初始化，避免初始bias导致崩溃
        for gate in self.fuse_gates:
            nn.init.zeros_(gate.bias)
            nn.init.normal_(gate.weight, mean=0.0, std=0.01)
        
        # 输出层 - 权重绑定（lm_head 复用 tok_embedding）
        self.norm = nn.LayerNorm(dim)
        
        self.to(DEVICE)
    
    def _init_states(self, batch_size, device, dtype):
        """为每层MHDSRA2初始化独立状态。
        
        Initialize independent state for each MHDSRA2 layer.
        每层维护自己的slot记忆，避免层间状态覆盖。
        """
        return [layer.init_state(batch_size, device=device, dtype=dtype) 
                for layer in self.mh_layers]

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """获取因果注意力掩码，支持缓存复用。

        Get causal attention mask with caching for reuse.
        缓存已生成过的掩码，避免重复计算。

        调用方 / Called by:
            self.forward
        被调用方 / Calls:
            nn.Transformer.generate_square_subsequent_mask
        参数 / Parameters:
            seq_len: 序列长度 / sequence length
            device: 设备 / device
        返回值 / Returns:
            torch.Tensor: 因果掩码 [seq_len, seq_len] / causal mask
        """
        key = (seq_len, device)
        if key not in self._causal_mask_cache:
            self._causal_mask_cache[key] = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )
        return self._causal_mask_cache[key]

    def forward(self, x, states=None, return_gate_info=False):
        """前向传播 — ST 与 MHDSRA2 并行处理，每层输出级融合。
        
        Forward pass — ST and MHDSRA2 process in parallel, fused at each layer output.
        每层MHDSRA2使用独立状态，避免层间slot记忆覆盖。
        
        调用方 / Called by:
            train_epoch, evaluate (via model(x))
        被调用方 / Calls:
            self.mh_layers, self.st_layers, self.st_projs, self.fuse_gates, self.norm
        参数 / Parameters:
            x: [batch, seq_len] 输入 token IDs / input token IDs
            states: List[MHDSRA2State] 或 None / list of per-layer states or None
            return_gate_info: 是否返回融合门控权重信息 / whether to return gate weight info
        返回值 / Returns:
            logits: [batch, seq_len, vocab_size] 输出 logits / output logits
            states: List[MHDSRA2State] 每层更新后的状态 / updated per-layer states
            gate_info: Dict[str, float] 门控权重统计（仅 return_gate_info=True 时返回）
        副作用 / Side effects:
            不修改输入states列表，返回新的states列表 / Does not mutate input states list, returns new list
        中文关键词:
            混合架构, 前向传播, 并行融合, 独立状态, ST分支, MHDSRA2分支, 分块处理, 门控权重
        English keywords:
            hybrid, forward, parallel_fusion, independent_state, ST, MHDSRA2, chunk, gate_weights
        """
        seq_len = x.shape[1]
        bsz = x.shape[0]
        
        # 初始化每层独立状态
        if states is None:
            states = self._init_states(bsz, x.device, x.dtype)
        
        # 创建新列表存储更新后的states，避免修改调用方传入的原始列表
        new_states = list(states)
        
        # 分块处理
        all_h = []
        gate_info = {} if return_gate_info else None
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk = x[:, start:end]
            chunk_len = end - start
            
            positions = torch.arange(chunk_len, device=x.device)
            h = self.tok_embedding(chunk) + self.pos_embedding(positions)
            
            # 逐层处理：ST和MHDSRA2并行，输出级融合
            for i, (mh_layer, st_layer, st_proj, fuse_gate, mh_out_norm, st_out_norm, mh_scale) in enumerate(
                zip(self.mh_layers, self.st_layers, self.st_projs, self.fuse_gates, self.mh_out_norms, self.st_out_norms, self.mh_scales)
            ):
                # ST分支 - 因果Transformer注意力
                causal_mask = self._get_causal_mask(chunk_len, x.device)
                h_st = st_layer(h, src_mask=causal_mask, is_causal=True)
                h_st = st_proj(h_st)
                # ST 分支也做 LayerNorm，与 MH 分支对称，消除幅度差异
                h_st = st_out_norm(h_st)
                
                # MHDSRA2分支 - 流式注意力
                mh_result = mh_layer(h, state=new_states[i], return_aux=True)
                if len(mh_result) == 3:
                    h_mh, new_states[i], _aux = mh_result
                else:
                    h_mh, new_states[i] = mh_result
                
                # 应用 LayerNorm 和缩放 - 平衡两个分支的输出幅度
                h_mh = mh_out_norm(h_mh) * mh_scale
                
                # 收集 slot 写入统计信息（用于监控 slot 是否正常写入）
                if return_gate_info and _aux is not None and "write_stats" in _aux and _aux["write_stats"] is not None:
                    with torch.no_grad():
                        ws = _aux["write_stats"]
                        gate_info[f"layer{i}_token_gate_mean"] = ws.get("token_gate_mean", 0.0)
                        gate_info[f"layer{i}_write_mass_mean"] = ws.get("write_mass_mean", 0.0)
                        gate_info[f"layer{i}_usage_mean"] = new_states[i].usage.mean().item()
                
                # 动态融合门控 - 基于两个分支的输出决定权重
                gate_input = torch.cat([h_st, h_mh], dim=-1)
                gate_logits = fuse_gate(gate_input)
                # 截断 logits 到 [-3, 3]，防止 softmax 过度极端（对应概率约 [0.0025, 0.9975]）
                gate_logits = gate_logits.clamp(-3.0, 3.0)
                gate_weights = F.softmax(gate_logits, dim=-1)
                
                # 收集门控权重统计信息（用于监控融合门控是否退化）
                if return_gate_info:
                    with torch.no_grad():
                        gate_info[f"layer{i}_st_weight"] = gate_weights[..., 0].mean().item()
                        gate_info[f"layer{i}_mh_weight"] = gate_weights[..., 1].mean().item()
                
                h = gate_weights[..., 0:1] * h_st + gate_weights[..., 1:2] * h_mh
            
            all_h.append(h)
        
        h = torch.cat(all_h, dim=1)
        h = self.norm(h)
        # 权重绑定：lm_head 复用 tok_embedding 的权重
        logits = F.linear(h, self.tok_embedding.weight)
        
        if return_gate_info:
            return logits, new_states, gate_info
        return logits, new_states


def train_epoch(model, dataloader, optimizer, scheduler, swanlab_run, warmup_steps, base_lr,
                global_step, scaler=None, grad_accum_steps=1, fuse_gate_frozen_steps=100, 
                gate_reg_weight=0.3, save_every_steps=0, output_dir=None, best_ppl=float('inf'),
                tokenizer=None, dim=256, n_layers=4, n_heads=4, slots=128, seq_len=512):
    """训练一个epoch。
    
    Train one epoch.
    
    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        swanlab_run: SwanLab运行实例
        warmup_steps: warmup步数
        base_lr: 基础学习率
        global_step: 全局步数
        scaler: 梯度缩放器（混合精度训练用）
        grad_accum_steps: 梯度累积步数
        fuse_gate_frozen_steps: 门控冻结步数（前N步不更新门控权重）
        gate_reg_weight: 门控正则化权重
        save_every_steps: 每N步自动保存检查点，0表示不按步保存
            / auto-save checkpoint every N steps, 0=disabled
        output_dir: 检查点输出目录 / checkpoint output directory
        best_ppl: 当前最佳PPL / current best PPL
        tokenizer: 分词器（用于保存vocab_size）/ tokenizer for saving vocab_size
        dim: 模型维度 / model dimension
        n_layers: 层数 / number of layers
        n_heads: 头数 / number of heads
        slots: slot数量 / number of slots
        seq_len: 序列长度 / sequence length
        
    Returns:
        total_loss / n_batches: 平均损失
        global_step: 更新后的全局步数
        best_ppl: 更新后的最佳PPL
    """
    if grad_accum_steps < 1:
        grad_accum_steps = 1
    
    model.train()
    total_loss = 0
    total_gate_reg_loss = 0
    n_batches = 0
    states = None
    
    # 进度日志到文件（绕过 swanlab 输出拦截）
    _log_f = open(Path(output_dir) / "train_progress.log", "a", encoding="utf-8") if output_dir else None
    
    optimizer.zero_grad()
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # Warmup阶段
        if global_step < warmup_steps:
            lr_scale = (global_step + 1) / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * lr_scale
        
        # 门控冻结：前N步不更新门控权重和 token_write_gate
        gate_frozen = global_step < fuse_gate_frozen_steps
        for gate in model.fuse_gates:
            for param in gate.parameters():
                param.requires_grad_(not gate_frozen)
        # 同时冻结 token_write_gate，防止训练初期写入概率被梯度压低
        for mh_layer in model.mh_layers:
            for param in mh_layer.token_write_gate.parameters():
                param.requires_grad_(not gate_frozen)
        
        # 混合精度前向传播
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=scaler is not None):
            logits, states, gate_info = model(x, states=states, return_gate_info=True)
            
            # 主损失
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            
            # 门控正则化损失：鼓励门控权重保持平衡（熵最大）
            gate_reg_loss = 0.0
            if gate_info and not gate_frozen:
                gate_reg_loss = 0.0
                for i in range(model.n_layers):
                    st_w = torch.tensor(gate_info[f"layer{i}_st_weight"], device=DEVICE, dtype=torch.float32)
                    mh_w = torch.tensor(gate_info[f"layer{i}_mh_weight"], device=DEVICE, dtype=torch.float32)
                    # 熵损失：0.693 = ln(2) 是最大熵（均匀分布）
                    entropy = - (st_w * torch.log(st_w + 1e-8) + mh_w * torch.log(mh_w + 1e-8))
                    gate_reg_loss += (0.693 - entropy)  # 鼓励熵大
                gate_reg_loss = gate_reg_loss / model.n_layers
            
            total_loss_step = ce_loss + gate_reg_loss * gate_reg_weight
            loss = total_loss_step / grad_accum_steps
        
        # 混合精度反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 梯度累积：只在accum_steps的倍数步执行优化
        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # MH scales 约束：防止被优化到极端值，保持两个分支平衡
            for scale_param in model.mh_scales:
                if scale_param.grad is not None:
                    scale_param.grad.data = scale_param.grad.data.clamp(min=-0.5, max=0.5)
            # 优化器步骤后强制 mh_scales 在 [0.5, 3.0] 范围内
            for scale_param in model.mh_scales:
                with torch.no_grad():
                    scale_param.data = scale_param.data.clamp(min=0.5, max=3.0)
            
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            
            # CosineAnnealing在optimizer.step()之后调用
            if global_step >= warmup_steps:
                scheduler.step()
        
        total_loss += ce_loss.item() if 'ce_loss' in locals() else loss.item() * grad_accum_steps
        if 'gate_reg_loss' in locals() and gate_reg_loss != 0:
            total_gate_reg_loss += gate_reg_loss.item()
        n_batches += 1
        global_step += 1
        
        # 记录到SwanLab
        if global_step % 200 == 0 and swanlab_run.enabled:
            current_lr = optimizer.param_groups[0]['lr']
            log_data = {
                "train/loss": ce_loss.item() if 'ce_loss' in locals() else loss.item() * grad_accum_steps,
                "train/ppl": math.exp(min(ce_loss.item() if 'ce_loss' in locals() else loss.item() * grad_accum_steps, 20)),
                "train/log_ppl": ce_loss.item() if 'ce_loss' in locals() else loss.item() * grad_accum_steps,
                "train/learning_rate": current_lr,
            }
            if 'gate_reg_loss' in locals() and gate_reg_loss != 0:
                log_data["train/gate_reg_loss"] = gate_reg_loss.item()
            log_data["train/gate_frozen"] = int(gate_frozen)
            # 记录融合门控权重和 slot 写入统计（监控是否退化）
            if gate_info:
                st_avg = 0.0
                mh_avg = 0.0
                token_gate_avg = 0.0
                write_mass_avg = 0.0
                usage_avg = 0.0
                st_count = 0
                layer_count = 0
                for k, v in gate_info.items():
                    log_data[f"gate/{k}"] = v
                    if "st_weight" in k:
                        st_avg += v
                        st_count += 1
                    elif "mh_weight" in k:
                        mh_avg += v
                    elif "token_gate_mean" in k:
                        token_gate_avg += v
                        layer_count += 1
                    elif "write_mass_mean" in k:
                        write_mass_avg += v
                    elif "usage_mean" in k:
                        usage_avg += v
                if st_count > 0:
                    st_avg /= st_count
                    mh_avg /= st_count
                    log_data["gate/avg_st_weight"] = st_avg
                    log_data["gate/avg_mh_weight"] = mh_avg
                    log_data["gate/avg_st_ratio"] = st_avg / (st_avg + mh_avg + 1e-8)
                # 记录 mh_scales 的值（监控是否被优化到太小）
                scales_values = [s.item() for s in model.mh_scales]
                log_data["mh_scale/avg"] = sum(scales_values) / len(scales_values)
                for i, s in enumerate(scales_values):
                    log_data[f"mh_scale/layer{i}"] = s
                if layer_count > 0:
                    token_gate_avg /= layer_count
                    write_mass_avg /= layer_count
                    usage_avg /= layer_count
                    log_data["slot/avg_token_gate"] = token_gate_avg
                    log_data["slot/avg_write_mass"] = write_mass_avg
                    log_data["slot/avg_usage"] = usage_avg
            swanlab_run.log(log_data, step=global_step)
            gate_str = ""
            slot_str = ""
            if gate_info:
                st_avg = sum(v for k, v in gate_info.items() if "st_weight" in k) / max(1, sum(1 for k in gate_info if "st_weight" in k))
                gate_str = f", Gate_ST_avg={st_avg:.3f}"
                if gate_frozen:
                    gate_str += " (FROZEN)"
                # 收集 slot 写入信息
                token_gate_list = [v for k, v in gate_info.items() if "token_gate_mean" in k]
                usage_list = [v for k, v in gate_info.items() if "usage_mean" in k]
                if token_gate_list:
                    slot_str = f", Slot_token_gate={sum(token_gate_list)/len(token_gate_list):.4f}, Slot_usage={sum(usage_list)/len(usage_list):.4f}"
            raw_loss = ce_loss.item() if 'ce_loss' in locals() else loss.item() * grad_accum_steps
            ppl_str = f"PPL={math.exp(min(raw_loss, 20)):.1f}" if raw_loss <= 20 else f"logPPL={raw_loss:.3f}"
            _msg = f"  Step {global_step}: Loss={raw_loss:.3f}, {ppl_str}, LR={current_lr:.6f}{gate_str}{slot_str}"
            if _log_f:
                _log_f.write(_msg + "\n")
                _log_f.flush()
        
        # 按步数自动保存检查点
        if save_every_steps > 0 and global_step % save_every_steps == 0 and output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            ckpt_path = f"{output_dir}/checkpoint_step{global_step}.pt"
            torch.save({
                "epoch": -1,
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_ppl": best_ppl,
                "vocab_size": len(tokenizer.encoder) if tokenizer is not None else 0,
                "config": {
                    "dim": dim, "n_layers": n_layers, "n_heads": n_heads,
                    "slots": slots, "seq_len": seq_len,
                },
            }, ckpt_path)
            _msg = f"  * 自动保存检查点 (Step {global_step})"
            if _log_f:
                _log_f.write(_msg + "\n")
                _log_f.flush()
            # 只保留最近2个step级检查点，删除更早的
            ckpt_files = sorted(Path(output_dir).glob("checkpoint_step*.pt"))
            if len(ckpt_files) > 2:
                for old_ckpt in ckpt_files[:-2]:
                    old_ckpt.unlink()
    
    if _log_f:
        _log_f.close()
    return total_loss / n_batches, global_step, best_ppl


def evaluate(model, dataloader, use_amp=True):
    """评估模型，跨batch传递MHDSRA2 states以利用slot记忆。

    Evaluate model with cross-batch MHDSRA2 state passing for slot memory.
    评估时shuffle=False，数据顺序排列，states传递天然合理。

    Args:
        model: 模型
        dataloader: 数据加载器
        use_amp: 是否使用混合精度训练

    Returns:
        avg_loss: 平均损失
        perplexity: 困惑度
    """
    model.eval()
    total_loss = 0
    n_batches = 0
    states = None

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(
                device_type='cuda',
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                enabled=use_amp
            ):
                logits, states = model(x, states=states)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity


def pretrain_hybrid_model(
    vocab_size=32000,
    dim=256,
    n_layers=4,
    n_heads=4,
    slots=128,
    seq_len=512,
    batch_size=16,
    lr=1e-3,
    max_steps=50000,
    warmup_steps=1000,
    max_epochs=30,
    data_dir="data/wikitext-103",
    tokenizer_path="models/bpe_tokenizer.json",
    output_dir="models/hybrid_lm",
    resume_from=None,
    use_amp=True,
    use_gradient_checkpointing=True,
    grad_accum_steps=1,
    fuse_gate_frozen_steps=100,
    gate_reg_weight=0.3,
    experiment_name=None,
):
    """预训练混合架构语言模型。
    
    Pre-train hybrid architecture language model.
    
    Args:
        vocab_size: 词汇表大小
        dim: 模型维度
        n_layers: MHDSRA2层数
        n_heads: 注意力头数
        slots: slot数量
        seq_len: 序列长度
        batch_size: batch大小
        lr: 学习率
        max_steps: 最大训练步数
        warmup_steps: warmup步数
        max_epochs: 最大epoch数
        data_dir: 数据目录
        tokenizer_path: 分词器路径
        output_dir: 输出目录
        resume_from: 恢复训练的检查点路径，None则从头训练
        use_amp: 是否使用混合精度训练
        use_gradient_checkpointing: 是否使用梯度检查点
        grad_accum_steps: 梯度累积步数
        fuse_gate_frozen_steps: 门控冻结步数（前N步不更新门控权重）
        gate_reg_weight: 门控正则化权重
        experiment_name: SwanLab实验名，None则使用默认名
    """
    print("="*70)
    print("混合架构语言模型预训练")
    print("="*70)
    
    set_seed(SEED)
    
    # 0. 初始化SwanLab
    dtype_str = "bf16" if torch.cuda.is_bf16_supported() else "fp16" if use_amp else "fp32"
    exp_name = experiment_name if experiment_name else "hybrid_lm_pretrain_v4_gate_fix"
    swanlab_run = init_swanlab(
        project="DSRA",
        experiment_name=exp_name,
        config={
            "vocab_size": vocab_size,
            "dim": dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "slots": slots,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "lr": lr,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "max_epochs": max_epochs,
            "resume_from": resume_from,
            "use_amp": use_amp,
            "dtype": dtype_str,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "grad_accum_steps": grad_accum_steps,
            "fuse_gate_frozen_steps": fuse_gate_frozen_steps,
            "gate_reg_weight": gate_reg_weight,
        },
        mode="cloud",
        description="混合架构语言模型预训练 v4（门控修复：初始化+正则化+冻结）",
    )
    
    # 1. 加载或训练分词器
    if Path(tokenizer_path).exists():
        print(f"加载已有分词器: {tokenizer_path}")
        tokenizer = FastBPETokenizer.load(tokenizer_path)
    else:
        print("训练新分词器...")
        data_paths = download_wikitext103(data_dir)
        train_text = load_text(data_paths["train"])
        tokenizer = FastBPETokenizer(vocab_size=vocab_size)
        tokenizer.train(train_text)
        tokenizer.save(tokenizer_path)
    
    # 2. 加载数据（使用完整WikiText-103训练集/验证集/测试集）
    print("\n加载数据...")
    data_paths = download_wikitext103(data_dir)
    train_text = load_text(data_paths["train"])  # 完整训练集：540M字符
    valid_text = load_text(data_paths["valid"])   # 完整验证集：1.1M字符
    test_text = load_text(data_paths["test"])     # 完整测试集：1.3M字符
    print(f"  训练集: {len(train_text):,} 字符")
    print(f"  验证集: {len(valid_text):,} 字符")
    print(f"  测试集: {len(test_text):,} 字符")
    
    train_dataset = StreamingDataset(tokenizer, train_text, seq_len=seq_len)
    valid_dataset = StreamingDataset(tokenizer, valid_text, seq_len=seq_len)
    test_dataset = StreamingDataset(tokenizer, test_text, seq_len=seq_len)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True
    )
    
    # 3. 创建模型
    print("\n创建模型...")
    model = HybridLanguageModel(
        vocab_size=len(tokenizer.encoder),
        dim=dim, n_layers=n_layers, n_heads=n_heads,
        slots=slots, chunk_size=seq_len,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {num_params:.2f}M")
    print(f"梯度检查点: {'✓' if use_gradient_checkpointing else '✗'}")
    print(f"混合精度训练: {'✓' if use_amp else '✗'}")
    if use_amp:
        print(f"  数据类型: {dtype_str}")
    
    # 4. 优化器和调度器 - 分组参数：embedding/LayerNorm 不做 weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'embedding' in name or 'norm' in name or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=max((max_steps - warmup_steps) // grad_accum_steps, 1), eta_min=1e-5)
    
    # 混合精度训练缩放器
    scaler = torch.amp.GradScaler('cuda', enabled=True) if use_amp else None
    
    # 5. 恢复训练
    global_step = 0
    best_ppl = float('inf')
    start_epoch = 0
    
    if resume_from and Path(resume_from).exists():
        print(f"\n从检查点恢复训练: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=DEVICE, weights_only=False)
        
        # 检查配置兼容性
        config_compatible = True
        if "config" in checkpoint:
            cp_config = checkpoint["config"]
            print(f"  检查点配置: {cp_config}")
            if cp_config.get("n_layers") != n_layers:
                print(f"  ❌ 检查点层数 {cp_config.get('n_layers')} 与当前配置 {n_layers} 不匹配！")
                config_compatible = False
            if cp_config.get("dim") != dim:
                print(f"  ❌ 检查点维度 {cp_config.get('dim')} 与当前配置 {dim} 不匹配！")
                config_compatible = False
            if cp_config.get("n_heads") != n_heads:
                print(f"  ❌ 检查点头数 {cp_config.get('n_heads')} 与当前配置 {n_heads} 不匹配！")
                config_compatible = False
        
        if config_compatible:
            # strict=False 兼容旧模型结构（如 st_transformer → st_layers 等变更）
            missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if missing:
                print(f"  ⚠ 缺少键（将使用随机初始化）: {len(missing)} 个")
            if unexpected:
                print(f"  ⚠ 多余键（将被忽略）: {len(unexpected)} 个")
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                # 兼容旧 checkpoint（未保存 scheduler_state_dict）：手动设置 LR 并重新开始余弦周期
                if global_step >= warmup_steps:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
            if "step" in checkpoint:
                global_step = checkpoint["step"]
            if "best_ppl" in checkpoint:
                best_ppl = checkpoint["best_ppl"]
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
            print(f"  恢复步数: {global_step:,}")
            print(f"  恢复epoch: {start_epoch}")
            print(f"  历史最佳PPL: {best_ppl:.1f}")
        else:
            print(f"  ⚠ 配置不兼容，跳过加载检查点，从头开始训练")
    
    # 6. 训练循环
    print("\n开始训练...")
    print(f"  Warmup步数: {warmup_steps:,}")
    print(f"  最大训练步数: {max_steps:,}")
    print(f"  最大epoch数: {max_epochs}")
    print(f"  基础学习率: {lr}")
    print(f"  梯度累积: {grad_accum_steps} 步")
    print(f"  起始epoch: {start_epoch + 1}")
    
    start_time = time.time()
    patience = 5
    patience_counter = 0
    
    for epoch in range(start_epoch, max_epochs):
        train_loss, global_step, best_ppl = train_epoch(
            model, train_loader, optimizer, scheduler, swanlab_run, 
            warmup_steps, lr, global_step,
            scaler=scaler,
            grad_accum_steps=grad_accum_steps,
            fuse_gate_frozen_steps=fuse_gate_frozen_steps,
            gate_reg_weight=gate_reg_weight,
            save_every_steps=2000,
            output_dir=output_dir,
            best_ppl=best_ppl,
            tokenizer=tokenizer,
            dim=dim, n_layers=n_layers, n_heads=n_heads,
            slots=slots, seq_len=seq_len,
        )
        
        # 评估
        valid_loss, valid_ppl = evaluate(model, valid_loader, use_amp=use_amp)
        print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.3f}, Valid PPL={valid_ppl:.1f}")
        
        # 记录到SwanLab
        if swanlab_run.enabled:
            swanlab_run.log({
                "valid/loss": valid_loss,
                "valid/ppl": valid_ppl,
                "train/epoch": epoch + 1,
                "train/avg_loss": train_loss,
            }, step=global_step)
        
        # 保存最佳模型
        if valid_ppl < best_ppl:
            best_ppl = valid_ppl
            patience_counter = 0
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_ppl": best_ppl,
                "vocab_size": len(tokenizer.encoder),
                "config": {
                    "dim": dim, "n_layers": n_layers, "n_heads": n_heads,
                    "slots": slots, "seq_len": seq_len,
                },
            }, f"{output_dir}/best_model.pt")
            print(f"  * 保存最佳模型 (PPL={best_ppl:.1f})")
            
            # 记录最佳模型到SwanLab
            if swanlab_run.enabled:
                swanlab_run.log({
                    "best_ppl": best_ppl,
                }, step=global_step)
        else:
            patience_counter += 1
            print(f"  PPL未改善 ({patience_counter}/{patience})")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_ppl": best_ppl,
                "vocab_size": len(tokenizer.encoder),
                "config": {
                    "dim": dim, "n_layers": n_layers, "n_heads": n_heads,
                    "slots": slots, "seq_len": seq_len,
                },
            }, f"{output_dir}/checkpoint_epoch{epoch+1}.pt")
            print(f"  * 保存检查点 (Epoch {epoch+1})")
        
        # 早停
        if global_step >= max_steps:
            print(f"\n达到最大训练步数 {max_steps:,}，停止训练")
            break
        
        if patience_counter >= patience:
            print(f"\nPPL连续{patience}个epoch未改善，提前停止")
            break
    
    elapsed = time.time() - start_time
    print(f"\n训练完成! 用时: {elapsed/3600:.1f}小时")
    print(f"最佳验证PPL: {best_ppl:.1f}")
    print(f"模型保存至: {output_dir}")
    
    # 训练结束后在测试集上评估最佳模型
    best_model_path = Path(output_dir) / "best_model.pt"
    if best_model_path.exists():
        print("\n在测试集上评估最佳模型...")
        checkpoint = torch.load(str(best_model_path), map_location=DEVICE, weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        test_loss, test_ppl = evaluate(model, test_loader, use_amp=use_amp)
        print(f"测试集 Loss: {test_loss:.4f}, 测试集 PPL: {test_ppl:.2f}")
        if swanlab_run.enabled:
            swanlab_run.log({
                "test/loss": test_loss,
                "test/ppl": test_ppl,
            }, step=global_step)
    else:
        print("\n⚠ 未找到最佳模型，跳过测试集评估")
    
    # 关闭SwanLab
    if swanlab_run.enabled:
        swanlab_run.finish()
    
    return model, tokenizer


if __name__ == "__main__":
    # 完整模型训练（使用完整WikiText-103训练集/验证集/测试集）
    # 注意：resume_from 设置为 None，确保从头开始训练新的 8 层模型
    model, tokenizer = pretrain_hybrid_model(
        vocab_size=32000,
        dim=256,
        n_layers=8,
        n_heads=8,
        slots=256,
        seq_len=512,
        batch_size=16,
        lr=2e-4,
        max_steps=100000,
        warmup_steps=5000,
        max_epochs=50,
        data_dir="data/wikitext-103",
        tokenizer_path="models/bpe_tokenizer.json",
        output_dir="models/hybrid_lm",
        resume_from=None,  # 确保从头开始训练新的 8 层模型
        use_amp=True,
        use_gradient_checkpointing=True,
        grad_accum_steps=4,
        fuse_gate_frozen_steps=100,
        gate_reg_weight=0.3,
    )
