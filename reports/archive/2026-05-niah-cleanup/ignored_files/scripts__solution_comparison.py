"""MHDSRA2 解决方案对比测试脚本

测试 5 个解决方案在 seq_len=2048 时的效果：
- 方案 1: 增加 Slot 数量
- 方案 2: 课程学习逐步训练
- 方案 3: 改进 Slot 写入策略
- 方案 4: 改进 Slot 读取策略
- 方案 5: 混合注意力机制

使用方式：
  python scripts/solution_comparison.py --test basic       # 方案 1-4
  python scripts/solution_comparison.py --test hybrid      # 方案 5
  python scripts/solution_comparison.py --test all         # 全部测试
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
import time

from src.dsra.mhdsra2.improved_dsra_mha import MultiHeadDSRA2, MHDSRA2Config


# ============================================================================
# 全局配置
# ============================================================================

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


# ============================================================================
# 数据生成器
# ============================================================================

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


# ============================================================================
# 模型包装
# ============================================================================

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


class StandardAttention(nn.Module):
    """Standard Transformer for comparison"""
    def __init__(self, vocab_size, dim=128, layers=2, heads=4, max_seq_len=4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.lm_head = nn.Linear(dim, vocab_size)
        self.to(DEVICE)
    
    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        h = self.embedding(x) + self.pos_embedding(positions)
        h = self.transformer(h)
        logits = self.lm_head(h[:, -1, :])
        return logits


# ============================================================================
# 训练和评估
# ============================================================================

def train_model(model, data_gen, vocab_size=VOCAB_SIZE, seq_len=2048, steps=2000, lr=1e-3, distance=128):
    """训练模型并返回准确率、loss 曲线和显存"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.empty_cache()
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_curve = []
    acc_curve = []
    
    log_interval = max(100, steps // 10)
    
    for step in range(steps):
        X, Y = data_gen(8, seq_len, vocab_size, distance=distance)
        
        if isinstance(model, MHDSRA2Wrapper):
            logits, _ = model(X)
        else:
            logits = model(X)
        
        loss = F.cross_entropy(logits, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        loss_curve.append(loss.item())
        
        if (step + 1) % log_interval == 0 or step == steps - 1:
            with torch.no_grad():
                acc = 0
                total = 0
                for _ in range(20):
                    Xt, Yt = data_gen(4, seq_len, vocab_size, distance=distance)
                    valid_mask = Yt != PAD
                    if valid_mask.sum() == 0:
                        continue
                    if isinstance(model, MHDSRA2Wrapper):
                        pred, _ = model(Xt)
                    else:
                        pred = model(Xt)
                    pred = pred.argmax(1)
                    acc += (pred[valid_mask] == Yt[valid_mask]).sum().item()
                    total += valid_mask.sum().item()
                if total > 0:
                    acc_curve.append(acc / total)
    
    final_acc = acc_curve[-1] if acc_curve else 0.0
    peak_memory = 0.0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
        torch.cuda.empty_cache()
    
    return final_acc, loss_curve, acc_curve, peak_memory


def train_curriculum(model, stages, lr=1e-3):
    """课程学习：分阶段训练
    
    Args:
        stages: list of (seq_len, steps) tuples
    """
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for stage_idx, (seq_len, steps) in enumerate(stages):
        if stage_idx > 0:
            lr *= 0.5
            for param_group in opt.param_groups:
                param_group['lr'] = lr
        
        for step in range(steps):
            X, Y = generate_niah(8, seq_len, VOCAB_SIZE, distance=128)
            if isinstance(model, MHDSRA2Wrapper):
                logits, _ = model(X)
            else:
                logits = model(X)
            loss = F.cross_entropy(logits, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    with torch.no_grad():
        acc = 0
        total = 0
        for _ in range(200):
            Xt, Yt = generate_niah(4, 2048, VOCAB_SIZE, distance=128)
            valid_mask = Yt != PAD
            if valid_mask.sum() == 0:
                continue
            if isinstance(model, MHDSRA2Wrapper):
                pred, _ = model(Xt)
            else:
                pred = model(Xt)
            pred = pred.argmax(1)
            acc += (pred[valid_mask] == Yt[valid_mask]).sum().item()
            total += valid_mask.sum().item()
    
    return acc / total if total > 0 else 0.0


# ============================================================================
# 方案测试函数
# ============================================================================

def test_solution_1():
    """方案 1: 增加 Slot 数量"""
    print(f"\n{'='*70}")
    print(f"方案 1: 增加 Slot 数量")
    print(f"{'='*70}")
    
    results = {}
    
    for slots in [64, 128, 256]:
        print(f"\n--- Testing slots={slots} ---")
        set_seed(SEED)
        cfg = MHDSRA2Config(
            dim=128, heads=4, local_window=64, slot_pe="rope",
            momentum_qkv=False, use_context_film=False, slots=slots,
            tau_init=16.0, tau_write_init=16.0,
        )
        model = MHDSRA2Wrapper(VOCAB_SIZE, cfg, num_layers=2)
        acc, _, _, mem = train_model(model, generate_niah, steps=2000)
        results[f"slots={slots}"] = {"acc": acc, "mem": mem}
        print(f"  Acc: {acc:.3f}, Memory: {mem:.1f} MB")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def test_solution_2():
    """方案 2: 课程学习"""
    print(f"\n{'='*70}")
    print(f"方案 2: 课程学习逐步训练")
    print(f"{'='*70}")
    
    results = {}
    
    set_seed(SEED)
    cfg = MHDSRA2Config(
        dim=128, heads=4, local_window=64, slot_pe="rope",
        momentum_qkv=False, use_context_film=False, slots=64,
        tau_init=16.0, tau_write_init=16.0,
    )
    model = MHDSRA2Wrapper(VOCAB_SIZE, cfg, num_layers=2)
    
    stages = [
        (512, 1000),
        (1024, 1500),
        (2048, 2000),
    ]
    
    start_time = time.time()
    acc = train_curriculum(model, stages)
    elapsed = time.time() - start_time
    
    results["curriculum"] = {"acc": acc, "time": elapsed}
    print(f"  Acc: {acc:.3f}, Time: {elapsed:.1f}s")
    
    return results


def test_solution_3():
    """方案 3: 改进 Slot 写入策略"""
    print(f"\n{'='*70}")
    print(f"方案 3: 改进 Slot 写入策略")
    print(f"{'='*70}")
    
    results = {}
    
    # 3a: 写入频率控制 (每 4 个 token 写入一次)
    configs = {
        "freq=4": {"write_frequency": 4, "novelty_threshold": 0.0, "write_protection": 0},
        "freq=4, threshold=0.3": {"write_frequency": 4, "novelty_threshold": 0.3, "write_protection": 0},
        "freq=4, threshold=0.5": {"write_frequency": 4, "novelty_threshold": 0.5, "write_protection": 0},
        "freq=4, threshold=0.3, protect=2": {"write_frequency": 4, "novelty_threshold": 0.3, "write_protection": 2},
    }
    
    for name, params in configs.items():
        print(f"\n--- Testing {name} ---")
        set_seed(SEED)
        cfg = MHDSRA2Config(
            dim=128, heads=4, local_window=64, slot_pe="rope",
            momentum_qkv=False, use_context_film=False, slots=64,
            tau_init=16.0, tau_write_init=16.0,
            write_frequency=params['write_frequency'],
            novelty_threshold=params['novelty_threshold'],
            write_protection=params['write_protection'],
        )
        model = MHDSRA2Wrapper(VOCAB_SIZE, cfg, num_layers=2)
        acc, _, _, mem = train_model(model, generate_niah, steps=2000)
        results[name] = {"acc": acc, "mem": mem}
        print(f"  Acc: {acc:.3f}, Memory: {mem:.1f} MB")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def test_solution_4():
    """方案 4: 改进 Slot 读取策略"""
    print(f"\n{'='*70}")
    print(f"方案 4: 改进 Slot 读取策略 (通过 tau_init 控制温度)")
    print(f"{'='*70}")
    
    results = {}
    
    # 方案 4: 通过降低读取温度 (tau_init) 来提高检索精度
    # tau 越小，注意力越锐利
    for tau in [16.0, 8.0, 4.0, 2.0]:
        print(f"\n--- Testing tau={tau} ---")
        set_seed(SEED)
        cfg = MHDSRA2Config(
            dim=128, heads=4, local_window=64, slot_pe="rope",
            momentum_qkv=False, use_context_film=False, slots=64,
            tau_init=tau, tau_write_init=16.0,
        )
        model = MHDSRA2Wrapper(VOCAB_SIZE, cfg, num_layers=2)
        acc, _, _, mem = train_model(model, generate_niah, steps=2000)
        results[f"tau={tau}"] = {"acc": acc, "mem": mem}
        print(f"  Acc: {acc:.3f}, Memory: {mem:.1f} MB")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


class HybridAttentionModel(nn.Module):
    """方案 5: 混合注意力模型
    
    Layer 1: Standard Transformer (Global Attention)
    Layer 2: MHDSRA2 (Slot + Local Attention)
    
    Layer 1 使用全局注意力捕获长程依赖（不受 local_window 限制）
    Layer 2 使用 slot 进行状态压缩和记忆
    """
    def __init__(self, vocab_size, cfg_mh, st_dim=128, st_layers=1, st_heads=4):
        super().__init__()
        # Layer 1: Standard Transformer
        self.st_embedding = nn.Embedding(vocab_size, st_dim)
        self.st_pos_embedding = nn.Embedding(4096, st_dim)
        st_encoder_layer = nn.TransformerEncoderLayer(
            d_model=st_dim, nhead=st_heads, dim_feedforward=st_dim * 4,
            batch_first=True, activation='gelu'
        )
        self.st_transformer = nn.TransformerEncoder(st_encoder_layer, num_layers=st_layers)
        self.st_proj = nn.Linear(st_dim, cfg_mh.dim)  # 投影到 MHDSRA2 维度
        
        # Layer 2: MHDSRA2
        self.mh_embedding = nn.Embedding(vocab_size, cfg_mh.dim)
        self.mh_layers = nn.ModuleList([
            MultiHeadDSRA2(cfg_mh) for _ in range(1)
        ])
        self.lm_head = nn.Linear(cfg_mh.dim, vocab_size)
        self.to(DEVICE)
    
    def forward(self, x):
        # Layer 1: Standard Transformer (Global)
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        h_st = self.st_embedding(x) + self.st_pos_embedding(positions)
        h_st = self.st_transformer(h_st)
        h_st = self.st_proj(h_st)
        
        # Layer 2: MHDSRA2 (Slot + Local)
        h_mh = self.mh_embedding(x)
        h = h_st + h_mh  # 融合两个分支
        
        state = None
        for layer in self.mh_layers:
            result = layer(h, state=state, return_aux=True)
            if len(result) == 3:
                h, state, aux = result
            else:
                h, state = result
        
        logits = self.lm_head(h[:, -1, :])
        return logits


def test_solution_5():
    """方案 5: 混合注意力机制"""
    print(f"\n{'='*70}")
    print(f"方案 5: 混合注意力机制 (Global Attention + Slot)")
    print(f"{'='*70}")
    
    results = {}
    
    # 5a: Layer 1=Global, Layer 2=Slot+Local
    print(f"\n--- Testing Hybrid: Global(1) + Slot+Local(1) ---")
    set_seed(SEED)
    cfg = MHDSRA2Config(
        dim=128, heads=4, local_window=64, slot_pe="rope",
        momentum_qkv=False, use_context_film=False, slots=64,
        tau_init=16.0, tau_write_init=16.0,
    )
    model = HybridAttentionModel(VOCAB_SIZE, cfg, st_dim=128, st_layers=1, st_heads=4)
    acc, _, _, mem = train_model_hybrid(model, generate_niah, steps=2000)
    results["hybrid_global_slot"] = {"acc": acc, "mem": mem}
    print(f"  Acc: {acc:.3f}, Memory: {mem:.1f} MB")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 5b: Layer 1=Global(2), Layer 2=Slot+Local(1)
    print(f"\n--- Testing Hybrid: Global(2) + Slot+Local(1) ---")
    set_seed(SEED)
    cfg = MHDSRA2Config(
        dim=128, heads=4, local_window=64, slot_pe="rope",
        momentum_qkv=False, use_context_film=False, slots=64,
        tau_init=16.0, tau_write_init=16.0,
    )
    model = HybridAttentionModel(VOCAB_SIZE, cfg, st_dim=128, st_layers=2, st_heads=4)
    acc, _, _, mem = train_model_hybrid(model, generate_niah, steps=2000)
    results["hybrid_global2_slot"] = {"acc": acc, "mem": mem}
    print(f"  Acc: {acc:.3f}, Memory: {mem:.1f} MB")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def train_model_hybrid(model, data_gen, vocab_size=VOCAB_SIZE, seq_len=2048, steps=2000, lr=1e-3, distance=128):
    """训练混合模型并返回准确率、loss 曲线和显存"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.empty_cache()
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_curve = []
    acc_curve = []
    
    log_interval = max(100, steps // 10)
    
    for step in range(steps):
        X, Y = data_gen(8, seq_len, vocab_size, distance=distance)
        
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        loss_curve.append(loss.item())
        
        if (step + 1) % log_interval == 0 or step == steps - 1:
            with torch.no_grad():
                acc = 0
                total = 0
                for _ in range(20):
                    Xt, Yt = data_gen(4, seq_len, vocab_size, distance=distance)
                    valid_mask = Yt != PAD
                    if valid_mask.sum() == 0:
                        continue
                    pred = model(Xt)
                    pred = pred.argmax(1)
                    acc += (pred[valid_mask] == Yt[valid_mask]).sum().item()
                    total += valid_mask.sum().item()
                if total > 0:
                    acc_curve.append(acc / total)
    
    final_acc = acc_curve[-1] if acc_curve else 0.0
    peak_memory = 0.0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
        torch.cuda.empty_cache()
    
    return final_acc, loss_curve, acc_curve, peak_memory


# ============================================================================
# 主函数
# ============================================================================

def run_all_basic():
    """运行方案 1-4"""
    all_results = {}
    
    all_results["方案 1: 增加 Slot 数量"] = test_solution_1()
    all_results["方案 2: 课程学习"] = test_solution_2()
    all_results["方案 3: 改进写入策略"] = test_solution_3()
    all_results["方案 4: 改进读取策略"] = test_solution_4()
    
    print(f"\n{'='*70}")
    print("方案对比测试结果")
    print(f"{'='*70}")
    
    for test_name, results in all_results.items():
        print(f"\n{test_name}:")
        for config, metrics in results.items():
            if "skipped" in metrics:
                print(f"  {config}: {metrics['skipped']}")
            elif "acc" in metrics:
                print(f"  {config}: Acc={metrics['acc']:.3f}, Memory={metrics.get('mem', 0):.1f} MB")
    
    return all_results


def run_all_hybrid():
    """运行方案 5"""
    return test_solution_5()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MHDSRA2 解决方案对比测试")
    parser.add_argument("--test", choices=["basic", "hybrid", "all"], default="all")
    args = parser.parse_args()
    
    print("MHDSRA2 解决方案对比测试")
    print(f"设备: {DEVICE}")
    print(f"Seq Len: 2048, Distance: 128, Vocab: {VOCAB_SIZE}")
    
    if args.test in ["basic", "all"]:
        run_all_basic()
    
    if args.test in ["hybrid", "all"]:
        run_all_hybrid()
