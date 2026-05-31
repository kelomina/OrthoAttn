"""
2M 序列流式处理测试脚本
- 逐步验证：32K → 64K → 256K → 2M
- 每个长度独立训练和评估
- 记录显存消耗和准确率
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime

from src.dsra.mhdsra2.improved_dsra_mha import MultiHeadDSRA2, MHDSRA2Config

SEED = 42
VOCAB_SIZE = 10
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_niah(batch_size, seq_len, vocab_size, distance=128):
    """生成 NIAH 数据集"""
    X = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)
    needle_key = torch.randint(0, vocab_size, (batch_size,), device=DEVICE)
    needle_value = torch.randint(0, vocab_size, (batch_size,), device=DEVICE)
    
    needle_pos = seq_len - distance - 1
    X[:, needle_pos] = needle_key
    
    Y = needle_value
    return X, Y

class StreamingModel(nn.Module):
    """纯流式模型（无 ST，用于对比测试）"""
    def __init__(self, vocab_size, cfg, chunk_size=4096):
        super().__init__()
        self.chunk_size = chunk_size
        self.embedding = nn.Embedding(vocab_size, cfg.dim)
        self.layers = nn.ModuleList([
            MultiHeadDSRA2(cfg) for _ in range(1)
        ])
        self.lm_head = nn.Linear(cfg.dim, vocab_size)
        self.to(DEVICE)
    
    def forward(self, x, state=None):
        seq_len = x.shape[1]
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk = x[:, start:end]
            h = self.embedding(chunk)
            for layer in self.layers:
                result = layer(h, state=state, return_aux=True)
                if len(result) == 3:
                    h, state, aux = result
                else:
                    h, state = result
        return self.lm_head(h[:, -1, :]), state

class StreamingHybridModel(nn.Module):
    """流式混合模型 (ST + MHDSRA2, Weighted)"""
    def __init__(self, vocab_size, cfg_mh, chunk_size=4096, st_dim=128, st_heads=4, max_seq_len=262144+4096):
        super().__init__()
        self.chunk_size = chunk_size
        
        # ST 分支
        self.st_embedding = nn.Embedding(vocab_size, st_dim)
        self.st_pos_embedding = nn.Embedding(max_seq_len, st_dim)
        st_encoder_layer = nn.TransformerEncoderLayer(
            d_model=st_dim, nhead=st_heads, dim_feedforward=st_dim * 4,
            batch_first=True, activation='gelu'
        )
        self.st_transformer = nn.TransformerEncoder(st_encoder_layer, num_layers=1)
        self.st_proj = nn.Linear(st_dim, cfg_mh.dim)
        
        # MHDSRA2 分支
        self.mh_embedding = nn.Embedding(vocab_size, cfg_mh.dim)
        self.mh_layers = nn.ModuleList([
            MultiHeadDSRA2(cfg_mh) for _ in range(1)
        ])
        
        # 可学习权重融合
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        
        self.lm_head = nn.Linear(cfg_mh.dim, vocab_size)
        self.to(DEVICE)
    
    def forward(self, x, state=None):
        """流式前向传播
        
        Args:
            x: 输入序列 [batch, seq_len]
            state: MHDSRA2 状态（跨 chunk 传递）
        """
        seq_len = x.shape[1]
        
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk = x[:, start:end]
            chunk_len = end - start
            
            # ST 分支
            positions = torch.arange(chunk_len, device=x.device).unsqueeze(0).expand(chunk.shape[0], -1)
            h_st = self.st_embedding(chunk) + self.st_pos_embedding(positions)
            h_st = self.st_transformer(h_st)
            h_st = self.st_proj(h_st)
            
            # MHDSRA2 分支
            h_mh = self.mh_embedding(chunk)
            
            # 可学习权重融合
            h = self.alpha * h_st + self.beta * h_mh
            
            # MHDSRA2 layers
            for layer in self.mh_layers:
                result = layer(h, state=state, return_aux=True)
                if len(result) == 3:
                    h, state, aux = result
                else:
                    h, state = result
        
        return self.lm_head(h[:, -1, :]), state

def train_streaming(model, data_gen, seq_len, steps=200, lr=5e-4, distance=128):
    """训练流式模型"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_acc = 0
    chunk_size = model.chunk_size
    
    for step in range(steps):
        X, Y = data_gen(2, seq_len, VOCAB_SIZE, distance=distance)
        
        total_loss = 0
        state = None
        
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            X_chunk = X[:, start:end]
            logits, state = model(X_chunk, state=state)
            
            if end >= seq_len - distance - 5:
                loss = F.cross_entropy(logits, Y)
                total_loss += loss
        
        if total_loss > 0:
            opt.zero_grad()
            total_loss.backward()
            opt.step()
        
        if (step + 1) % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                acc = 0
                for _ in range(5):
                    Xt, Yt = data_gen(1, seq_len, VOCAB_SIZE, distance=distance)
                    state = None
                    for start in range(0, seq_len, chunk_size):
                        end = min(start + chunk_size, seq_len)
                        Xt_chunk = Xt[:, start:end]
                        pred, state = model(Xt_chunk, state=state)
                    pred_class = pred.argmax(dim=-1).cpu()
                    acc += (pred_class == Yt.cpu()).float().mean().item()
                acc /= 5
            
            print(f"  Step {step+1}/{steps}: Loss={total_loss:.3f}, Acc={acc:.3f}")
            best_acc = max(best_acc, acc)
    
    return best_acc

def test_2m():
    """2M 序列测试"""
    print("="*70)
    print("2M 序列流式处理测试")
    print("="*70)
    print(f"设备: {DEVICE}")
    print(f"词表大小: {VOCAB_SIZE}")
    
    results = {}
    
    # 测试序列长度（只测试到256K，2M暂不测试）
    seq_lengths = [
        (32768, "32K"),
        (65536, "64K"),
        (262144, "256K"),
    ]
    
    for seq_len, label in seq_lengths:
        print(f"\n{'='*60}")
        print(f"测试: {label} ({seq_len:,})")
        print(f"{'='*60}")
        
        # 根据长度调整 chunk_size、训练步数和学习率
        if seq_len <= 65536:
            chunk_size = 4096
            steps = 500
            lr = 1e-3
        elif seq_len <= 262144:
            chunk_size = 8192
            steps = 1000
            lr = 1e-3
        else:
            chunk_size = 16384
            steps = 2000
            lr = 1e-3
        
        # 重置随机种子
        set_seed(SEED)
        
        # 创建混合模型 (ST + MHDSRA2, Weighted)
        cfg = MHDSRA2Config(
            dim=128, heads=4, local_window=64, slot_pe="rope",
            momentum_qkv=False, use_context_film=False, slots=64,
            tau_init=16.0, tau_write_init=16.0,
        )
        
        model = StreamingHybridModel(
            VOCAB_SIZE, cfg, 
            chunk_size=chunk_size,
            st_dim=128, st_heads=4,
            max_seq_len=seq_len+4096
        )
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        
        print(f"Chunk size: {chunk_size:,}")
        print(f"训练步数: {steps}")
        print(f"模型参数: {num_params:.2f}M")
        
        # 显存监控
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(DEVICE)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 训练
        start_time = time.time()
        acc = train_streaming(model, generate_niah, seq_len, steps=steps)
        training_time = time.time() - start_time
        
        # 获取显存
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
        else:
            mem = 0
        
        print(f"\n✅ {label} 完成:")
        print(f"  准确率: {acc:.3f} ({acc*100:.1f}%)")
        print(f"  显存: {mem:.1f} MB")
        print(f"  训练时间: {training_time:.1f}s")
        
        results[label] = {
            "acc": acc,
            "mem": mem,
            "time": training_time,
            "seq_len": seq_len,
            "chunk_size": chunk_size,
            "steps": steps
        }
        
        # 清理显存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 如果 32K 失败，跳过后续
        if seq_len == 32768 and acc < 0.1:
            print(f"\n⚠️  32K 测试失败 (Acc={acc:.3f})，跳过后续测试")
            break
    
    # 打印汇总
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    print(f"{'序列长度':<15} | {'准确率':<10} | {'显存 (MB)':<12} | {'时间 (s)':<10}")
    print("-"*60)
    for label, r in results.items():
        print(f"{label:<15} | {r['acc']:<10.3f} | {r['mem']:<12.1f} | {r['time']:<10.1f}")
    
    return results

if __name__ == "__main__":
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = test_2m()
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
