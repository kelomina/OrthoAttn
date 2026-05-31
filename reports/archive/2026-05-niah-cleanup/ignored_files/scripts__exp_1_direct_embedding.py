"""实验 1: 直接使用 Embedding 作为 Key/Value（简化版）

根因假设：QKV 投影破坏了 key/value 的语义对应关系。
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
import torch
import torch.nn.functional as F


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

def generate_niah_data(batch_size, seq_len, vocab_size):
    PAD, QUERY, KEY, FILLER = 0, 1, 2, 4
    X = torch.randint(FILLER, vocab_size, (batch_size, seq_len))
    Y = torch.full((batch_size,), PAD, dtype=torch.long)
    for i in range(batch_size):
        needle_pos = random.randint(0, seq_len // 2)
        q_pos = needle_pos + 3
        if q_pos >= seq_len:
            continue
        val = random.randint(FILLER, vocab_size - 1)
        X[i, needle_pos] = KEY
        X[i, needle_pos + 1] = val
        X[i, needle_pos + 2] = KEY
        X[i, q_pos] = QUERY
        Y[i] = val
    return X, Y


class SimpleDirectKVModel(torch.nn.Module):
    """简化的直接 KV 模型：只使用一个全局 slot"""
    def __init__(self, vocab_size, dim=128, slots=64):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, dim)
        self.slot_k = torch.nn.Parameter(torch.randn(slots, dim) / dim**0.5)
        self.slot_v = torch.nn.Parameter(torch.randn(slots, dim) / dim**0.5)
        self.out = torch.nn.Linear(dim, vocab_size)
        self.slots = slots
        self.dim = dim
    
    def forward(self, x, state=None, ret_all=False):
        b, t = x.shape
        e = self.emb(x)  # [B,T,D]
        
        if state is None:
            state = {
                'sk': self.slot_k.data.clone(),
                'sv': self.slot_v.data.clone()
            }
        
        sk, sv = state['sk'], state['sv']
        
        # Write: 每个 token 写入一个 slot
        en = F.normalize(e, dim=-1)  # [B,T,D]
        skn = F.normalize(sk, dim=-1)  # [S,D]
        
        # [B,T,D] @ [D,S] -> [B,T,S]
        sim = en @ skn.T  # [B,T,D] @ [D,S] -> [B,T,S]
        _, top_idx = sim.max(dim=2)  # [B,T]
        
        # 更新 slot: 使用 last-write-wins (不使用 inplace)
        sv_new = sv.clone()
        for i in range(b):
            for j in range(t):
                slot_i = top_idx[i, j].item()
                sv_new[slot_i] = (0.9 * sv[slot_i] + 0.1 * e[i, j])
                sv_new[slot_i] = F.normalize(sv_new[slot_i].unsqueeze(0), dim=-1).squeeze(0)
        
        # Read: query 读取
        q = e[:, -1, :]  # [B,D] - 使用最后一个位置作为 query
        qn = F.normalize(q, dim=-1)  # [B,D]
        
        # [B,D] @ [D,S] -> [B,S]
        read_sim = qn @ F.normalize(sv_new, dim=-1).T  # [B,S]
        _, read_idx = read_sim.max(dim=1)  # [B]
        
        # 读取对应的 value
        out = torch.zeros(b, self.dim, device=x.device)
        for i in range(b):
            out[i] = sv[read_idx[i]]
        
        out = self.out(out)  # [B,V]
        
        state = {'sk': sk, 'sv': sv}
        if ret_all:
            return out.unsqueeze(1), state
        return out, state


class AttentionBasedModel(torch.nn.Module):
    """使用标准 attention 机制的模型作为对比"""
    def __init__(self, vocab_size, dim=128):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, dim)
        self.attn = torch.nn.MultiheadAttention(dim, 4, batch_first=True)
        self.out = torch.nn.Linear(dim, vocab_size)
    
    def forward(self, x, state=None, ret_all=False):
        e = self.emb(x)  # [B,T,D]
        # Self attention
        out, _ = self.attn(e, e, e)  # [B,T,D]
        out = self.out(out[:, -1, :])  # [B,V]
        if ret_all:
            return out.unsqueeze(1), state
        return out, state


def test():
    print("="*70)
    print("实验 1: 直接使用 Embedding 作为 Key/Value")
    print("="*70)
    
    vocab_size, seq_len, steps = 10, 128, 500
    results = {}
    
    # 1. 简化的直接 KV 模型
    print("\n--- Simple Direct KV Model ---")
    set_seed(42)
    model = SimpleDirectKVModel(vocab_size, dim=128, slots=64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for s in range(steps):
        X, Y = generate_niah_data(8, seq_len, vocab_size)
        logits, _ = model(X, ret_all=True)
        logits = logits.squeeze(1) if logits.dim() > 2 else logits
        loss = F.cross_entropy(logits, Y)
        opt.zero_grad(); loss.backward(); opt.step()
        
        if (s+1) % 100 == 0:
            with torch.no_grad():
                c, t = 0, 0
                for _ in range(20):
                    Xt, Yt = generate_niah_data(4, seq_len, vocab_size)
                    l, _ = model(Xt)
                    c += (l.argmax(1) == Yt).sum().item()
                    t += Yt.size(0)
                print(f"Step {s+1}/{steps} | Loss: {loss:.3f} | Acc: {c/t:.3f}")
    
    with torch.no_grad():
        c, t = 0, 0
        for _ in range(100):
            Xt, Yt = generate_niah_data(4, seq_len, vocab_size)
            l, _ = model(Xt)
            c += (l.argmax(1) == Yt).sum().item()
            t += Yt.size(0)
        acc = c/t
        print(f"\n最终 Test Acc: {acc:.3f}")
        results["Simple Direct KV"] = acc
    
    # 2. 标准 Attention 模型作为对比
    print("\n--- Standard Attention (对比基线) ---")
    set_seed(42)
    model2 = AttentionBasedModel(vocab_size, dim=128)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    
    for s in range(steps):
        X, Y = generate_niah_data(8, seq_len, vocab_size)
        logits, _ = model2(X, ret_all=True)
        logits = logits.squeeze(1) if logits.dim() > 2 else logits
        loss = F.cross_entropy(logits, Y)
        opt2.zero_grad(); loss.backward(); opt2.step()
        
        if (s+1) % 100 == 0:
            with torch.no_grad():
                c, t = 0, 0
                for _ in range(20):
                    Xt, Yt = generate_niah_data(4, seq_len, vocab_size)
                    l, _ = model2(Xt)
                    c += (l.argmax(1) == Yt).sum().item()
                    t += Yt.size(0)
                print(f"Step {s+1}/{steps} | Loss: {loss:.3f} | Acc: {c/t:.3f}")
    
    with torch.no_grad():
        c, t = 0, 0
        for _ in range(100):
            Xt, Yt = generate_niah_data(4, seq_len, vocab_size)
            l, _ = model2(Xt)
            c += (l.argmax(1) == Yt).sum().item()
            t += Yt.size(0)
        acc2 = c/t
        print(f"\n最终 Test Acc: {acc2:.3f}")
        results["Standard Attention"] = acc2
    
    print("\n" + "="*70)
    print("结果汇总:")
    for n, a in results.items():
        print(f"  {n}: {a:.3f}")
    return results


if __name__ == "__main__":
    test()
