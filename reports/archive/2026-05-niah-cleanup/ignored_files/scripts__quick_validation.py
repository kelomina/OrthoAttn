"""快速验证测试：测试最有希望的 3 个方案

方案 1: slots=256 (基础测试显示有效)
方案 3: write_frequency=4, threshold=0.3 (新实现)
方案 5: Hybrid Global+Slot (新实现)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dsra.mhdsra2.improved_dsra_mha import MultiHeadDSRA2, MHDSRA2Config


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 42
VOCAB_SIZE = 10
PAD = 0
QUERY = 1
KEY = 2
FILLER_START = 4


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def generate_niah(batch_size, seq_len, vocab_size, distance=128):
    """生成 NIAH 数据"""
    X = torch.randint(FILLER_START, vocab_size, (batch_size, seq_len))
    Y = torch.full((batch_size,), PAD, dtype=torch.long)
    
    needle_pos = 0
    query_pos = 2 + distance
    
    for i in range(batch_size):
        if query_pos >= seq_len:
            continue
        
        key = random.randint(FILLER_START, vocab_size - 1)
        value = random.randint(FILLER_START, vocab_size - 1)
        
        X[i, needle_pos] = KEY
        X[i, needle_pos + 1] = key
        X[i, needle_pos + 2] = value
        
        X[i, query_pos] = KEY
        X[i, query_pos + 1] = key
        X[i, query_pos + 2] = QUERY
        Y[i] = value
    
    return X.to(DEVICE), Y.to(DEVICE)


class MHDSRA2Wrapper(nn.Module):
    def __init__(self, vocab_size, cfg, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.dim)
        self.layers = nn.ModuleList([
            MultiHeadDSRA2(cfg) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(cfg.dim, vocab_size)
        self.to(DEVICE)
    
    def forward(self, x):
        h = self.embedding(x)
        state = None
        for layer in self.layers:
            result = layer(h, state=state, return_aux=True)
            if len(result) == 3:
                h, state, aux = result
            else:
                h, state = result
        logits = self.lm_head(h[:, -1, :])
        return logits, state


class HybridModel(nn.Module):
    """方案 5: 混合注意力模型"""
    def __init__(self, vocab_size, cfg_mh, st_dim=128, st_layers=1, st_heads=4):
        super().__init__()
        self.st_embedding = nn.Embedding(vocab_size, st_dim)
        self.st_pos_embedding = nn.Embedding(4096, st_dim)
        st_encoder_layer = nn.TransformerEncoderLayer(
            d_model=st_dim, nhead=st_heads, dim_feedforward=st_dim * 4,
            batch_first=True, activation='gelu'
        )
        self.st_transformer = nn.TransformerEncoder(st_encoder_layer, num_layers=st_layers)
        self.st_proj = nn.Linear(st_dim, cfg_mh.dim)
        
        self.mh_embedding = nn.Embedding(vocab_size, cfg_mh.dim)
        self.mh_layers = nn.ModuleList([
            MultiHeadDSRA2(cfg_mh) for _ in range(1)
        ])
        self.lm_head = nn.Linear(cfg_mh.dim, vocab_size)
        self.to(DEVICE)
    
    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        h_st = self.st_embedding(x) + self.st_pos_embedding(positions)
        h_st = self.st_transformer(h_st)
        h_st = self.st_proj(h_st)
        
        h_mh = self.mh_embedding(x)
        h = h_st + h_mh
        
        state = None
        for layer in self.mh_layers:
            result = layer(h, state=state, return_aux=True)
            if len(result) == 3:
                h, state, aux = result
            else:
                h, state = result
        
        logits = self.lm_head(h[:, -1, :])
        return logits


def train_and_evaluate(name, model, steps=2000, seq_len=2048, lr=1e-3):
    """训练并评估模型"""
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    set_seed(SEED)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.empty_cache()
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    distance = 128
    
    for step in range(steps):
        X, Y = generate_niah(8, seq_len, VOCAB_SIZE, distance=distance)
        
        if isinstance(model, HybridModel):
            logits = model(X)
        else:
            logits, _ = model(X)
        
        loss = F.cross_entropy(logits, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if (step + 1) % 200 == 0 or step == steps - 1:
            with torch.no_grad():
                acc = 0
                total = 0
                for _ in range(20):
                    Xt, Yt = generate_niah(4, seq_len, VOCAB_SIZE, distance=distance)
                    valid_mask = Yt != PAD
                    if valid_mask.sum() == 0:
                        continue
                    
                    if isinstance(model, HybridModel):
                        pred = model(Xt)
                    else:
                        pred, _ = model(Xt)
                    pred = pred.argmax(1)
                    acc += (pred[valid_mask] == Yt[valid_mask]).sum().item()
                    total += valid_mask.sum().item()
                if total > 0:
                    print(f"Step {step+1}/{steps}: Loss={loss.item():.3f}, Acc={acc/total:.3f}")
    
    peak_mem = 0.0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
        torch.cuda.empty_cache()
    
    return peak_mem


if __name__ == "__main__":
    print("MHDSRA2 快速验证测试")
    print(f"设备: {DEVICE}")
    print(f"Seq Len: 2048, Distance: 128\n")
    
    # 方案 1: slots=256
    print("方案 1: slots=256")
    cfg1 = MHDSRA2Config(
        dim=128, heads=4, local_window=64, slot_pe="rope",
        momentum_qkv=False, use_context_film=False, slots=256,
        tau_init=16.0, tau_write_init=16.0,
    )
    model1 = MHDSRA2Wrapper(VOCAB_SIZE, cfg1, num_layers=2)
    mem1 = train_and_evaluate("方案 1: slots=256", model1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 方案 3: 改进写入策略
    print("\n方案 3: 改进写入策略 (freq=4, threshold=0.3, protect=2)")
    cfg3 = MHDSRA2Config(
        dim=128, heads=4, local_window=64, slot_pe="rope",
        momentum_qkv=False, use_context_film=False, slots=64,
        tau_init=16.0, tau_write_init=16.0,
        write_frequency=4, novelty_threshold=0.3, write_protection=2,
    )
    model3 = MHDSRA2Wrapper(VOCAB_SIZE, cfg3, num_layers=2)
    mem3 = train_and_evaluate("方案 3: 改进写入策略", model3)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 方案 5: 混合注意力
    print("\n方案 5: 混合注意力 (Global + Slot)")
    cfg5 = MHDSRA2Config(
        dim=128, heads=4, local_window=64, slot_pe="rope",
        momentum_qkv=False, use_context_film=False, slots=64,
        tau_init=16.0, tau_write_init=16.0,
    )
    model5 = HybridModel(VOCAB_SIZE, cfg5, st_dim=128, st_layers=1, st_heads=4)
    mem5 = train_and_evaluate("方案 5: 混合注意力", model5)
    
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    print(f"{'方案':<30} | {'显存 (MB)':<12}")
    print("-" * 45)
    print(f"{'方案 1 (slots=256)':<30} | {mem1:<12.1f}")
    print(f"{'方案 3 (写入策略优化)':<30} | {mem3:<12.1f}")
    print(f"{'方案 5 (混合注意力)':<30} | {mem5:<12.1f}")
    print("="*70)
    print("\n注意：准确率已在训练过程中输出")
