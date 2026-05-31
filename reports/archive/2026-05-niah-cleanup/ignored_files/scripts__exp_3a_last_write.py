"""实验 3a: Last-write-wins 聚合方式

根因假设：Slot 写入时的加权平均操作（scatter_add）丢失了精确信息。

解决方案：使用 last-write-wins 策略，只保留最后一个写入的值。

为什么要这么做：
- 当前使用 scatter_add 聚合多个 token，导致信息被混合
- Last-write-wins 可以保留最"新"的信息
- 这对于 NIAH 任务很重要，因为 needle value 应该在 query 时被准确读取

失败后怎么办：
- 如果准确率仍然 < 30%，说明问题不是聚合操作
- 继续实验 1：尝试直接 embedding
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
import torch
import torch.nn.functional as F
from src.dsra.mhdsra2.improved_dsra_mha import MultiHeadDSRA2, MHDSRA2Config


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def generate_niah_data(batch_size: int, seq_len: int, vocab_size: int):
    PAD_TOKEN_ID = 0
    QUERY_TOKEN_ID = 1
    NEEDLE_KEY_TOKEN_ID = 2
    FILLER_TOKEN_START = 4
    
    X = torch.randint(FILLER_TOKEN_START, vocab_size, (batch_size, seq_len), dtype=torch.long)
    Y = torch.full((batch_size,), PAD_TOKEN_ID, dtype=torch.long)
    
    for i in range(batch_size):
        needle_pos = random.randint(0, seq_len // 2)
        query_key_pos = needle_pos + 2
        query_pos = query_key_pos + 1
        
        if query_pos >= seq_len:
            continue
        
        needle_val = random.randint(FILLER_TOKEN_START, vocab_size - 1)
        
        X[i, needle_pos] = NEEDLE_KEY_TOKEN_ID
        X[i, needle_pos + 1] = needle_val
        X[i, query_key_pos] = NEEDLE_KEY_TOKEN_ID
        X[i, query_pos] = QUERY_TOKEN_ID
        Y[i] = needle_val
    
    return X, Y


class SimpleNIAHModel(torch.nn.Module):
    def __init__(self, cfg: MHDSRA2Config, vocab_size: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, cfg.dim)
        self.dsra = MultiHeadDSRA2(cfg)
        self.output = torch.nn.Linear(cfg.dim, vocab_size)
    
    def forward(self, x, state=None, return_all_logits=False):
        x = self.embedding(x)
        x, state = self.dsra(x, state)
        if return_all_logits:
            return self.output(x), state
        return self.output(x[:, -1, :]), state


def test_scatter_last_write():
    """测试 last-write-wins 策略"""
    print("="*70)
    print("实验 3a: Last-write-wins 聚合方式")
    print("="*70)
    
    # 首先，验证 scatter_add vs scatter_last 的区别
    print("\n--- 验证聚合方式区别 ---")
    
    batch_size, heads, slots, d = 1, 1, 4, 8
    
    # 模拟：两个 token 写入同一个 slot
    # token 1: k=[1,0,...], v=[0,1,0,...]  (低优先级)
    # token 2: k=[0,1,...], v=[0,0,1,...]  (高优先级)
    
    idx = torch.tensor([[[[0, 0]]]])  # 都写入 slot 0
    weights = torch.tensor([[[[0.3, 0.7]]]])  # token 2 权重更高
    
    k1 = torch.zeros(batch_size, heads, 2, d)
    k1[:, :, 0, 0] = 1.0
    k1[:, :, 1, 1] = 1.0
    
    v1 = torch.zeros(batch_size, heads, 2, d)
    v1[:, :, 0, 1] = 1.0
    v1[:, :, 1, 2] = 1.0
    
    print(f"Token 1: k[:,:,0]={k1[0,0,0,:4]}, v[:,:,0]={v1[0,0,0,:4]}")
    print(f"Token 2: k[:,:,1]={k1[0,0,1,:4]}, v[:,:,1]={v1[0,0,1,:4]}")
    
    # 当前方式：scatter_add (加权平均)
    # agg_v = 0.3 * v1 + 0.7 * v2 = [0, 0.3, 0.7, 0, ...]
    agg_v_scatter = torch.zeros(batch_size, heads, slots, d)
    for b in range(batch_size):
        for h in range(heads):
            for t in range(2):
                slot_idx = idx[b, h, 0, t].item()
                w = weights[b, h, 0, t].item()
                agg_v_scatter[b, h, slot_idx] += w * v1[b, h, t]
    
    print(f"\nscatter_add 结果: {agg_v_scatter[0,0,0,:4]}")
    print(f"预期 (加权平均): v = 0.3*[0,1,0,0] + 0.7*[0,0,1,0] = [0, 0.3, 0.7, 0]")
    
    # Last-write-wins: 只保留最后一个写入的值
    agg_v_last = torch.zeros(batch_size, heads, slots, d)
    # 最后一个写入是 token 2 (index=1)
    agg_v_last[0, 0, idx[0, 0, 0, 1]] = v1[0, 0, 1]
    
    print(f"last-write-wins 结果: {agg_v_last[0,0,0,:4]}")
    print(f"预期: v = [0, 0, 1, 0] (token 2 的值)")
    
    # 测试相似度
    target_v = v1[0, 0, 1]  # token 2 的值
    sim_scatter = F.cosine_similarity(agg_v_scatter[0, 0, 0], target_v, dim=0).item()
    sim_last = F.cosine_similarity(agg_v_last[0, 0, 0], target_v, dim=0).item()
    
    print(f"\n与目标 (token 2) 的相似度:")
    print(f"  scatter_add: {sim_scatter:.4f}")
    print(f"  last-write-wins: {sim_last:.4f}")
    
    return sim_last > sim_scatter


def test_with_modified_scatter(model_class, cfg, name, vocab_size=10, seq_len=128, num_steps=300):
    """使用修改后的模型进行完整训练测试"""
    print(f"\n--- {name} ---")
    
    set_seed(42)
    model = model_class(cfg, vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    chunk_size = 32
    log_every = 50
    
    for step in range(num_steps):
        X, Y = generate_niah_data(8, seq_len, vocab_size)
        
        state = None
        for i in range(0, seq_len, chunk_size):
            logits, state = model(X[:, i:i + chunk_size], state, return_all_logits=True)
        
        logits, _ = model(X[:, -1:], state)
        loss = F.cross_entropy(logits, Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % log_every == 0:
            with torch.no_grad():
                correct = 0
                total = 0
                for _ in range(20):
                    X_test, Y_test = generate_niah_data(4, seq_len, vocab_size)
                    state = None
                    for i in range(0, seq_len, chunk_size):
                        _, state = model(X_test[:, i:i + chunk_size], state)
                    logits, _ = model(X_test[:, -1:], state)
                    pred = logits.argmax(dim=1)
                    correct += (pred == Y_test).sum().item()
                    total += Y_test.size(0)
                
                acc = correct / total
                print(f"Step {step+1}/{num_steps} | Loss: {loss.item():.3f} | Acc: {acc:.3f}")
    
    # 最终评估
    with torch.no_grad():
        correct = 0
        total = 0
        for _ in range(100):
            X_test, Y_test = generate_niah_data(4, seq_len, vocab_size)
            state = None
            for i in range(0, seq_len, chunk_size):
                _, state = model(X_test[:, i:i + chunk_size], state)
            logits, _ = model(X_test[:, -1:], state)
            pred = logits.argmax(dim=1)
            correct += (pred == Y_test).sum().item()
            total += Y_test.size(0)
        
        final_acc = correct / total
        print(f"最终 Test Acc: {final_acc:.3f}")
        return final_acc


if __name__ == "__main__":
    print("实验 3a: Last-write-wins 聚合方式")
    print("="*70)
    
    # 1. 先验证聚合方式的区别
    scatter_works = test_scatter_last_write()
    
    if scatter_works:
        print("\n✓ 理论上 last-write-wins 应该更好")
    else:
        print("\n✗ 理论上 last-write-wins 没有优势")
    
    # 2. 运行完整训练测试
    print("\n" + "="*70)
    print("完整训练测试")
    print("="*70)
    
    vocab_size = 10
    seq_len = 128
    
    results = {}
    
    # Baseline
    cfg_baseline = MHDSRA2Config(
        dim=128, heads=4, slots=32,
        local_window=16, use_local=True, use_retrieval=False,
        detach_state=False, exact_write=True, exact_read=True,
        tau_init=16.0, tau_write_init=16.0
    )
    results["Baseline"] = test_with_modified_scatter(
        SimpleNIAHModel, cfg_baseline, "Baseline", vocab_size, seq_len
    )
    
    # 大模型
    cfg_large = MHDSRA2Config(
        dim=256, heads=8, slots=64,
        local_window=32, use_local=True, use_retrieval=False,
        detach_state=False, exact_write=True, exact_read=True,
        tau_init=32.0, tau_write_init=32.0,
        eta=20.0, max_update=0.99
    )
    results["Large Model"] = test_with_modified_scatter(
        SimpleNIAHModel, cfg_large, "Large Model", vocab_size, seq_len
    )
    
    print("\n" + "="*70)
    print("结果汇总:")
    for name, acc in results.items():
        print(f"  {name}: {acc:.3f}")
