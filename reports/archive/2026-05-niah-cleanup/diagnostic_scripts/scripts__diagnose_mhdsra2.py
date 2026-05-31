"""MHDSRA2 训练诊断脚本

诊断 MHDSRA2 在长序列 (seq_len=2048) 时的训练崩溃问题。

输出:
- 梯度范数变化
- Slot 写入统计（write_gate, novelty, conflict）
- Fuse Gate 分布
- 不同学习率/步数的对比
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


# ============================================================================
# 配置
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
# MHDSRA2 Wrapper
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


def get_mhdsra2_config(local_window=64, dim=128, heads=4, slots=64):
    return MHDSRA2Config(
        dim=dim,
        heads=heads,
        local_window=local_window,
        slot_pe="rope",
        momentum_qkv=False,
        use_context_film=False,
        slots=slots,
        tau_init=16.0,
        tau_write_init=16.0,
    )


# ============================================================================
# 诊断函数
# ============================================================================

def diagnose_gradient_flow(seq_len=2048, lr=1e-3, steps=1000):
    """诊断梯度流动问题
    
    输出:
    - 每 100 步的 loss
    - 每 100 步的梯度范数（embedding, layers, lm_head）
    - slot 写入统计
    """
    print(f"\n{'='*70}")
    print(f"诊断 1: 梯度流动 (seq_len={seq_len}, lr={lr}, steps={steps})")
    print(f"{'='*70}")
    
    set_seed(SEED)
    cfg = get_mhdsra2_config(local_window=64)
    model = MHDSRA2Wrapper(VOCAB_SIZE, cfg, num_layers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    distance = 128
    
    for step in range(steps):
        X, Y = generate_niah(8, seq_len, VOCAB_SIZE, distance=distance)
        
        logits, state = model(X)
        loss = F.cross_entropy(logits, Y)
        
        opt.zero_grad()
        loss.backward()
        
        # 记录梯度范数
        if (step + 1) % 100 == 0 or step == 0:
            grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.norm().item()
            
            total_grad_norm = sum(v for v in grad_norms.values()) ** 0.5
            
            # Slot 统计
            layer0 = model.layers[0]
            write_stats = getattr(layer0, 'last_write_stats', None)
            
            print(f"\nStep {step+1}/{steps}")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Total Grad Norm: {total_grad_norm:.4f}")
            print(f"  Embedding Grad Norm: {grad_norms.get('embedding.weight', 0):.4f}")
            print(f"  Layer 0 Grad Norm: {sum(v for k, v in grad_norms.items() if 'layers.0' in k):.4f}")
            print(f"  LM Head Grad Norm: {grad_norms.get('lm_head.weight', 0):.4f}")
            
            if write_stats:
                print(f"  Slot Write Gate Mean: {write_stats['write_gate_mean']:.4f}")
                print(f"  Slot Novelty Mean: {write_stats['novelty_mean']:.4f}")
                print(f"  Slot Conflict Mean: {write_stats['conflict_mean']:.4f}")
            
            # 检查是否有 NaN 梯度
            has_nan = any(torch.isnan(param.grad).any() for param in model.parameters() if param.grad is not None)
            if has_nan:
                print(f"  ⚠️ NaN GRADIENTS DETECTED!")
        
        opt.step()
        
        # 评估准确率
        if (step + 1) % 200 == 0 or step == steps - 1:
            with torch.no_grad():
                acc = 0
                total = 0
                for _ in range(20):
                    Xt, Yt = generate_niah(4, seq_len, VOCAB_SIZE, distance=distance)
                    valid_mask = Yt != PAD
                    if valid_mask.sum() == 0:
                        continue
                    pred, _ = model(Xt)
                    pred = pred.argmax(1)
                    acc += (pred[valid_mask] == Yt[valid_mask]).sum().item()
                    total += valid_mask.sum().item()
                if total > 0:
                    print(f"  Accuracy: {acc/total:.3f}")


def diagnose_learning_rate(seq_len=2048, steps=2000):
    """对比不同学习率的效果"""
    print(f"\n{'='*70}")
    print(f"诊断 2: 学习率对比 (seq_len={seq_len})")
    print(f"{'='*70}")
    
    lrs = [1e-4, 5e-4, 1e-3, 5e-3]
    distance = 128
    
    for lr in lrs:
        print(f"\n--- Testing lr={lr} ---")
        set_seed(SEED)
        cfg = get_mhdsra2_config(local_window=64)
        model = MHDSRA2Wrapper(VOCAB_SIZE, cfg, num_layers=2)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        
        best_acc = 0
        for step in range(steps):
            X, Y = generate_niah(8, seq_len, VOCAB_SIZE, distance=distance)
            logits, _ = model(X)
            loss = F.cross_entropy(logits, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if (step + 1) % 200 == 0:
                with torch.no_grad():
                    acc = 0
                    total = 0
                    for _ in range(20):
                        Xt, Yt = generate_niah(4, seq_len, VOCAB_SIZE, distance=distance)
                        valid_mask = Yt != PAD
                        if valid_mask.sum() == 0:
                            continue
                        pred, _ = model(Xt)
                        pred = pred.argmax(1)
                        acc += (pred[valid_mask] == Yt[valid_mask]).sum().item()
                        total += valid_mask.sum().item()
                    if total > 0:
                        curr_acc = acc / total
                        best_acc = max(best_acc, curr_acc)
                        if curr_acc > 0.9:
                            print(f"  Step {step+1}: Acc={curr_acc:.3f} ✅")
                            break
                        else:
                            print(f"  Step {step+1}: Acc={curr_acc:.3f}")
        
        print(f"  Best Acc: {best_acc:.3f}")


def diagnose_local_window(seq_len=2048, lr=1e-3, steps=2000):
    """对比不同 local_window 的效果"""
    print(f"\n{'='*70}")
    print(f"诊断 3: Local Window 对比 (seq_len={seq_len})")
    print(f"{'='*70}")
    
    windows = [32, 64, 128, 256]
    distance = 128
    
    for w in windows:
        print(f"\n--- Testing local_window={w} ---")
        set_seed(SEED)
        cfg = get_mhdsra2_config(local_window=w)
        model = MHDSRA2Wrapper(VOCAB_SIZE, cfg, num_layers=2)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        
        best_acc = 0
        for step in range(steps):
            X, Y = generate_niah(8, seq_len, VOCAB_SIZE, distance=distance)
            logits, _ = model(X)
            loss = F.cross_entropy(logits, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if (step + 1) % 200 == 0:
                with torch.no_grad():
                    acc = 0
                    total = 0
                    for _ in range(20):
                        Xt, Yt = generate_niah(4, seq_len, VOCAB_SIZE, distance=distance)
                        valid_mask = Yt != PAD
                        if valid_mask.sum() == 0:
                            continue
                        pred, _ = model(Xt)
                        pred = pred.argmax(1)
                        acc += (pred[valid_mask] == Yt[valid_mask]).sum().item()
                        total += valid_mask.sum().item()
                    if total > 0:
                        curr_acc = acc / total
                        best_acc = max(best_acc, curr_acc)
                        if curr_acc > 0.9:
                            print(f"  Step {step+1}: Acc={curr_acc:.3f} ✅")
                            break
                        else:
                            print(f"  Step {step+1}: Acc={curr_acc:.3f}")
        
        print(f"  Best Acc: {best_acc:.3f}")


# ============================================================================
# 对比测试
# ============================================================================

def run_comparison_test(seq_len=2048, steps=1000):
    """对比 4 个方案的训练效果
    
    方案:
    - 基准: conflict_protection=0.3, RoPE=enabled
    - 方案A: conflict_protection=0.1, RoPE=enabled
    - 方案B: conflict_protection=0.3, RoPE=disabled
    - 方案AB: conflict_protection=0.1, RoPE=disabled
    """
    print(f"\n{'='*70}")
    print(f"对比测试: MHDSRA2 在 seq_len={seq_len} 的训练效果")
    print(f"{'='*70}")
    
    configs = {
        "基准": {"conflict_protection": 0.3, "slot_pe": "rope"},
        "方案A (冲突保护↓)": {"conflict_protection": 0.1, "slot_pe": "rope"},
        "方案B (关闭RoPE)": {"conflict_protection": 0.3, "slot_pe": "none"},
        "方案AB (双修改)": {"conflict_protection": 0.1, "slot_pe": "none"},
    }
    
    distance = 128
    
    for name, params in configs.items():
        print(f"\n{'='*60}")
        print(f"测试: {name}")
        print(f"  conflict_protection={params['conflict_protection']}, slot_pe={params['slot_pe']}")
        print(f"{'='*60}")
        
        set_seed(SEED)
        cfg = MHDSRA2Config(
            dim=128,
            heads=4,
            local_window=64,
            slot_pe=params['slot_pe'],
            momentum_qkv=False,
            use_context_film=False,
            slots=64,
            tau_init=16.0,
            tau_write_init=16.0,
            conflict_protection_coef=params['conflict_protection'],
        )
        model = MHDSRA2Wrapper(VOCAB_SIZE, cfg, num_layers=2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        best_acc = 0
        for step in range(steps):
            X, Y = generate_niah(8, seq_len, VOCAB_SIZE, distance=distance)
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
                        pred, _ = model(Xt)
                        pred = pred.argmax(1)
                        acc += (pred[valid_mask] == Yt[valid_mask]).sum().item()
                        total += valid_mask.sum().item()
                    if total > 0:
                        curr_acc = acc / total
                        best_acc = max(best_acc, curr_acc)
                        print(f"  Step {step+1}/{steps}: Loss={loss.item():.3f}, Acc={curr_acc:.3f}")
        
        print(f"  ✅ Best Acc: {best_acc:.3f}")
    
    print(f"\n{'='*70}")
    print(f"对比测试完成")
    print(f"{'='*70}")


def run_slot_ablation_test(seq_len=2048, steps=1500):
    """Slot 消融测试：禁用 Slot 输出，只使用 Local Attention
    
    目的：验证是否是 slot 机制导致长序列训练崩溃
    """
    print(f"\n{'='*70}")
    print(f"Slot 消融测试: seq_len={seq_len}")
    print(f"{'='*70}")
    print("目的：验证是否是 slot 机制导致训练崩溃")
    
    distance = 128
    configs = {
        "完整 MHDSRA2": {"slot_weight": 1.0, "local_weight": 0.0, "retrieval_weight": 0.0},
        "仅 Local Attention": {"slot_weight": 0.0, "local_weight": 1.0, "retrieval_weight": 0.0},
        "仅 Slot": {"slot_weight": 1.0, "local_weight": 0.0, "retrieval_weight": 0.0, "zero_local": True},
    }
    
    for name, params in configs.items():
        print(f"\n{'='*60}")
        print(f"测试: {name}")
        print(f"  slot_weight={params['slot_weight']}, local_weight={params['local_weight']}")
        print(f"{'='*60}")
        
        set_seed(SEED)
        cfg = MHDSRA2Config(
            dim=128,
            heads=4,
            local_window=64,
            slot_pe="rope",
            momentum_qkv=False,
            use_context_film=False,
            slots=64,
            tau_init=16.0,
            tau_write_init=16.0,
            conflict_protection_coef=0.3,
        )
        
        model = MHDSRA2Wrapper(VOCAB_SIZE, cfg, num_layers=2)
        
        # 对于"仅 Local Attention"测试，修改 fuse_gate 权重
        if params.get('slot_weight', 1.0) == 0.0:
            # 强制 fuse_gate 输出偏向 local (index 1)
            # fuse_gate 是 Linear(d_head, 3)，输出 logits 为 [slot_gate, local_gate, retrieval_gate]
            # 设置 bias 使 local_gate >> slot_gate, retrieval_gate
            for layer in model.layers:
                device = layer.fuse_gate.bias.device
                layer.fuse_gate.bias.data = torch.tensor([-10.0, 10.0, -10.0], device=device)
        
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        best_acc = 0
        for step in range(steps):
            X, Y = generate_niah(8, seq_len, VOCAB_SIZE, distance=distance)
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
                        pred, _ = model(Xt)
                        pred = pred.argmax(1)
                        acc += (pred[valid_mask] == Yt[valid_mask]).sum().item()
                        total += valid_mask.sum().item()
                    if total > 0:
                        curr_acc = acc / total
                        best_acc = max(best_acc, curr_acc)
                        print(f"  Step {step+1}/{steps}: Loss={loss.item():.3f}, Acc={curr_acc:.3f}")
        
        print(f"  ✅ Best Acc: {best_acc:.3f}")
    
    print(f"\n{'='*70}")
    print(f"Slot 消融测试完成")
    print(f"{'='*70}")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("MHDSRA2 长序列训练诊断")
    print(f"设备: {DEVICE}")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["gradient", "lr", "window", "comparison", "ablation", "all"], default="ablation")
    args = parser.parse_args()
    
    if args.test == "ablation":
        run_slot_ablation_test()
    elif args.test == "comparison":
        run_comparison_test()
    elif args.test == "gradient":
        diagnose_gradient_flow()
    elif args.test == "lr":
        diagnose_learning_rate()
    elif args.test == "window":
        diagnose_local_window()
    elif args.test == "all":
        run_comparison_test()
        diagnose_gradient_flow()
        run_slot_ablation_test()
        diagnose_learning_rate()
        diagnose_local_window()
