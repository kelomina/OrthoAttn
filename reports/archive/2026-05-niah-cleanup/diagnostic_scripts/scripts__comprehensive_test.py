"""MHDSRA2 混合架构优化与全面测试脚本

Phase 1: 混合架构优化（3种配置）
Phase 2: 消融测试
Phase 3: 2M 序列测试
Phase 4: 真实场景验证

使用方式：
  python scripts/comprehensive_test.py --phase 1    # Phase 1
  python scripts/comprehensive_test.py --phase 2    # Phase 2
  python scripts/comprehensive_test.py --phase 3    # Phase 3
  python scripts/comprehensive_test.py --phase 4    # Phase 4
  python scripts/comprehensive_test.py --phase all  # 全部测试
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


def generate_document_qa(batch_size, seq_len, vocab_size, num_paragraphs=5):
    """生成文档问答模拟数据 (Phase 4)"""
    X = torch.randint(FILLER_START, vocab_size, (batch_size, seq_len))
    Y = torch.full((batch_size,), PAD, dtype=torch.long)
    
    para_len = max(20, (seq_len - 100) // num_paragraphs)
    
    for i in range(batch_size):
        target_para = random.randint(0, num_paragraphs - 1)
        target_key = random.randint(FILLER_START, vocab_size - 1)
        target_value = random.randint(FILLER_START, vocab_size - 1)
        
        current_pos = 0
        for p in range(num_paragraphs):
            if current_pos + para_len >= seq_len:
                break
            
            if p == target_para:
                X[i, current_pos] = KEY
                X[i, current_pos + 1] = target_key
                X[i, current_pos + 2] = target_value
                for j in range(3, para_len):
                    if current_pos + j < seq_len:
                        X[i, current_pos + j] = target_value
            
            current_pos += para_len
        
        query_pos = seq_len - 10
        if query_pos > current_pos:
            X[i, query_pos] = KEY
            X[i, query_pos + 1] = target_key
            X[i, query_pos + 2] = QUERY
            Y[i] = target_value
    
    return X.to(DEVICE), Y.to(DEVICE)


def generate_code_understanding(batch_size, seq_len, vocab_size, num_vars=8):
    """生成代码理解模拟数据 (Phase 4)"""
    X = torch.randint(FILLER_START, vocab_size, (batch_size, seq_len))
    Y = torch.full((batch_size,), PAD, dtype=torch.long)
    
    var_spacing = max(16, (seq_len - 100) // num_vars)
    var_values = {}
    
    for i in range(batch_size):
        var_values = {}
        current_pos = 0
        
        for v in range(num_vars):
            if current_pos + 3 >= seq_len:
                break
            
            var_id = v + FILLER_START
            value = random.randint(FILLER_START, vocab_size - 1)
            var_values[var_id] = value
            
            X[i, current_pos] = KEY
            X[i, current_pos + 1] = var_id
            X[i, current_pos + 2] = value
            
            current_pos += var_spacing
        
        query_var = random.choice(list(var_values.keys()))
        query_pos = min(current_pos + 2, seq_len - 4)
        
        if query_pos + 3 < seq_len:
            X[i, query_pos] = KEY
            X[i, query_pos + 1] = query_var
            X[i, query_pos + 2] = QUERY
            Y[i] = var_values[query_var]
    
    return X.to(DEVICE), Y.to(DEVICE)


def generate_info_update(batch_size, seq_len, vocab_size, num_updates=3):
    """生成信息更新模拟数据 (Phase 4)"""
    X = torch.randint(FILLER_START, vocab_size, (batch_size, seq_len))
    Y = torch.full((batch_size,), PAD, dtype=torch.long)
    
    entity_id = random.randint(FILLER_START, vocab_size - 1)
    update_spacing = max(20, (seq_len - 100) // (num_updates + 1))
    
    for i in range(batch_size):
        current_pos = 0
        latest_value = None
        
        for u in range(num_updates):
            if current_pos + 3 >= seq_len:
                break
            
            value = random.randint(FILLER_START, vocab_size - 1)
            latest_value = value
            
            X[i, current_pos] = KEY
            X[i, current_pos + 1] = entity_id
            X[i, current_pos + 2] = value
            
            current_pos += update_spacing
        
        query_pos = min(current_pos + 2, seq_len - 4)
        if query_pos + 3 < seq_len and latest_value is not None:
            X[i, query_pos] = KEY
            X[i, query_pos + 1] = entity_id
            X[i, query_pos + 2] = QUERY
            Y[i] = latest_value
    
    return X.to(DEVICE), Y.to(DEVICE)


# ============================================================================
# 混合架构模型
# ============================================================================

class HybridModel(nn.Module):
    """混合注意力模型 (Phase 1 基础版)"""
    def __init__(self, vocab_size, cfg_mh, st_dim=128, st_layers=1, st_heads=4, fusion_type='sum', max_seq_len=65536):
        super().__init__()
        self.st_embedding = nn.Embedding(vocab_size, st_dim)
        self.st_pos_embedding = nn.Embedding(max_seq_len, st_dim)
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
        
        self.fusion_type = fusion_type
        if fusion_type == 'weighted':
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.beta = nn.Parameter(torch.tensor(0.5))
        
        if fusion_type == 'residual':
            self.ln = nn.LayerNorm(cfg_mh.dim)
        
        self.lm_head = nn.Linear(cfg_mh.dim, vocab_size)
        self.to(DEVICE)
    
    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        h_st = self.st_embedding(x) + self.st_pos_embedding(positions)
        h_st = self.st_transformer(h_st)
        h_st = self.st_proj(h_st)
        
        h_mh = self.mh_embedding(x)
        
        if self.fusion_type == 'weighted':
            h = self.alpha * h_st + self.beta * h_mh
        elif self.fusion_type == 'residual':
            h = self.ln(h_st + h_mh)
        else:  # sum
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


class StreamingHybridModel(nn.Module):
    """流式混合模型 (Phase 3)"""
    def __init__(self, vocab_size, cfg_mh, chunk_size=1024, st_dim=128, st_heads=4, max_seq_len=2097152+4096):
        super().__init__()
        self.chunk_size = chunk_size
        
        self.st_embedding = nn.Embedding(vocab_size, st_dim)
        self.st_pos_embedding = nn.Embedding(max_seq_len, st_dim)
        st_encoder_layer = nn.TransformerEncoderLayer(
            d_model=st_dim, nhead=st_heads, dim_feedforward=st_dim * 4,
            batch_first=True, activation='gelu'
        )
        self.st_transformer = nn.TransformerEncoder(st_encoder_layer, num_layers=1)
        self.st_proj = nn.Linear(st_dim, cfg_mh.dim)
        
        self.mh_embedding = nn.Embedding(vocab_size, cfg_mh.dim)
        self.mh_layers = nn.ModuleList([
            MultiHeadDSRA2(cfg_mh) for _ in range(1)
        ])
        
        self.lm_head = nn.Linear(cfg_mh.dim, vocab_size)
        self.to(DEVICE)
    
    def forward(self, x, state=None):
        """流式前向传播
        
        Args:
            x: 输入序列 [batch, seq_len]
            state: MHDSRA2 状态（跨 chunk 传递）
        """
        seq_len = x.shape[1]
        all_logits = []
        
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk = x[:, start:end]
            chunk_len = end - start
            
            positions = torch.arange(chunk_len, device=x.device).unsqueeze(0).expand(chunk.shape[0], -1)
            h_st = self.st_embedding(chunk) + self.st_pos_embedding(positions)
            h_st = self.st_transformer(h_st)
            h_st = self.st_proj(h_st)
            
            h_mh = self.mh_embedding(chunk)
            h = h_st + h_mh
            
            for layer in self.mh_layers:
                result = layer(h, state=state, return_aux=True)
                if len(result) == 3:
                    h, state, aux = result
                else:
                    h, state = result
            
            all_logits.append(self.lm_head(h[:, -1, :]))
        
        return torch.stack(all_logits, dim=1), state


# ============================================================================
# 训练和评估
# ============================================================================

def train_and_evaluate(name, model, data_gen, steps=2000, seq_len=2048, lr=1e-3, distance=128, pass_distance=False, eval_interval=None, eval_batches=20, **data_gen_kwargs):
    """训练并评估模型
    
    Args:
        name: 测试名称
        model: 模型
        data_gen: 数据生成器
        steps: 训练步数
        seq_len: 序列长度
        lr: 学习率
        distance: NIAH距离参数
        pass_distance: 是否将distance传递给data_gen
        eval_interval: 评估间隔步数（默认steps//10）
        eval_batches: 评估批次数量（默认20）
        **data_gen_kwargs: 数据生成器的其他参数
    """
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    set_seed(SEED)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.empty_cache()
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_acc = 0
    
    log_interval = eval_interval if eval_interval else max(100, steps // 10)
    
    def call_data_gen(batch_size, seq_len, vocab_size, **kwargs):
        """智能调用数据生成器"""
        call_kwargs = {**data_gen_kwargs, **kwargs}
        if pass_distance:
            call_kwargs['distance'] = distance
        return data_gen(batch_size, seq_len, vocab_size, **call_kwargs)
    
    for step in range(steps):
        if isinstance(model, StreamingHybridModel):
            X, Y = call_data_gen(4, seq_len, VOCAB_SIZE)
            chunk_size = model.chunk_size
            
            total_loss = 0
            num_chunks = 0
            state = None
            
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                if end >= seq_len - distance - 5:
                    X_chunk = X[:, :end]
                    logits, state = model(X_chunk, state=state)
                    if isinstance(logits, torch.Tensor) and logits.dim() == 3:
                        loss = F.cross_entropy(logits[:, -1, :], Y)
                    else:
                        loss = F.cross_entropy(logits, Y)
                    total_loss += loss
                    num_chunks += 1
                    break
        else:
            X, Y = call_data_gen(8, seq_len, VOCAB_SIZE)
            logits = model(X)
            loss = F.cross_entropy(logits, Y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if (step + 1) % log_interval == 0 or step == steps - 1:
            with torch.no_grad():
                acc = 0
                total = 0
                for _ in range(eval_batches):
                    if isinstance(model, StreamingHybridModel):
                        Xt, Yt = call_data_gen(2, seq_len, VOCAB_SIZE)
                        chunk_size = model.chunk_size
                        state = None
                        for start in range(0, seq_len, chunk_size):
                            end = min(start + chunk_size, seq_len)
                            if end >= seq_len - distance - 5:
                                Xt_chunk = Xt[:, :end]
                                pred, state = model(Xt_chunk, state=state)
                                if isinstance(pred, torch.Tensor) and pred.dim() == 3:
                                    pred = pred[:, -1, :]
                                break
                    else:
                        Xt, Yt = call_data_gen(4, seq_len, VOCAB_SIZE)
                        pred = model(Xt)
                    
                    valid_mask = Yt != PAD
                    if valid_mask.sum() == 0:
                        continue
                    pred = pred.argmax(1)
                    acc += (pred[valid_mask] == Yt[valid_mask]).sum().item()
                    total += valid_mask.sum().item()
                if total > 0:
                    curr_acc = acc / total
                    best_acc = max(best_acc, curr_acc)
                    print(f"Step {step+1}/{steps}: Loss={loss.item():.3f}, Acc={curr_acc:.3f}")
    
    peak_mem = 0.0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
        torch.cuda.empty_cache()
    
    print(f"\nFinal Best Acc: {best_acc:.3f}, Peak Memory: {peak_mem:.1f} MB")
    return best_acc, peak_mem


# ============================================================================
# Phase 1: 混合架构优化
# ============================================================================

def phase1_architecture_optimization():
    """Phase 1: 混合架构优化"""
    print("\n" + "="*70)
    print("Phase 1: 混合架构优化")
    print("="*70)
    
    results = {}
    
    configs = [
        {"name": "Baseline (1 ST + 1 MH)", "st_layers": 1, "fusion": "sum"},
        {"name": "A (2 ST + 1 MH)", "st_layers": 2, "fusion": "sum"},
        {"name": "B (1 ST + 1 MH, Weighted)", "st_layers": 1, "fusion": "weighted"},
        {"name": "C (2 ST + 1 MH, Residual)", "st_layers": 2, "fusion": "residual"},
    ]
    
    for cfg in configs:
        set_seed(SEED)
        mh_cfg = MHDSRA2Config(
            dim=128, heads=4, local_window=64, slot_pe="rope",
            momentum_qkv=False, use_context_film=False, slots=64,
            tau_init=16.0, tau_write_init=16.0,
        )
        
        model = HybridModel(
            VOCAB_SIZE, mh_cfg, 
            st_dim=128, st_layers=cfg["st_layers"], st_heads=4,
            fusion_type=cfg["fusion"]
        )
        
        acc, mem = train_and_evaluate(
            cfg["name"], model, generate_niah, 
            steps=2000, seq_len=2048, pass_distance=True,
            eval_interval=100, eval_batches=10
        )
        results[cfg["name"]] = {"acc": acc, "mem": mem}
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# ============================================================================
# Phase 2: 消融测试
# ============================================================================

def phase2_ablation_test():
    """Phase 2: 消融测试"""
    print("\n" + "="*70)
    print("Phase 2: 消融测试")
    print("="*70)
    
    results = {}
    
    # 纯 ST (2 层) - 扩展位置编码
    class PureST(nn.Module):
        def __init__(self, vocab_size, dim=128, layers=2, heads=4, max_seq_len=32768):
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
            return self.lm_head(h[:, -1, :])
    
    print("\n--- Pure ST (2 layers) ---")
    set_seed(SEED)
    model_st = PureST(VOCAB_SIZE, dim=128, layers=2, heads=4)
    acc, mem = train_and_evaluate("Pure ST", model_st, generate_niah, steps=1500, seq_len=2048, pass_distance=True, eval_interval=100, eval_batches=10)
    results["Pure ST (2 layers)"] = {"acc": acc, "mem": mem}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 纯 MHDSRA2 (2 层)
    class PureMH(nn.Module):
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
            return self.lm_head(h[:, -1, :])
    
    print("\n--- Pure MHDSRA2 (2 layers) ---")
    set_seed(SEED)
    mh_cfg = MHDSRA2Config(
        dim=128, heads=4, local_window=64, slot_pe="rope",
        momentum_qkv=False, use_context_film=False, slots=64,
        tau_init=16.0, tau_write_init=16.0,
    )
    model_mh = PureMH(VOCAB_SIZE, mh_cfg, num_layers=2)
    acc, mem = train_and_evaluate("Pure MHDSRA2", model_mh, generate_niah, steps=2000, seq_len=2048, pass_distance=True, eval_interval=100, eval_batches=10)
    results["Pure MHDSRA2 (2 layers)"] = {"acc": acc, "mem": mem}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 混合架构 (最优) - Phase 1 方案B
    print("\n--- Hybrid (1 ST + 1 MH, Weighted) - Phase 1 Optimal ---")
    set_seed(SEED)
    model_hybrid = HybridModel(
        VOCAB_SIZE, mh_cfg, 
        st_dim=128, st_layers=1, st_heads=4,
        fusion_type="weighted"
    )
    acc, mem = train_and_evaluate("Hybrid", model_hybrid, generate_niah, steps=2000, seq_len=2048, pass_distance=True, eval_interval=100, eval_batches=10)
    results["Hybrid (1 ST + 1 MH, Weighted)"] = {"acc": acc, "mem": mem}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


# ============================================================================
# Phase 3: 2M 序列测试
# ============================================================================

def phase3_streaming_test():
    """Phase 3: 2M 序列测试（流式处理）"""
    print("\n" + "="*70)
    print("Phase 3: 2M 序列测试（流式处理）")
    print("="*70)
    
    results = {}
    
    # 逐步测试：32K → 64K → 256K → 2M
    seq_lengths = [32768, 65536, 262144, 2097152]
    distance = 128
    
    for sl in seq_lengths:
        print(f"\n{'='*60}")
        print(f"Testing seq_len={sl:,} ({sl/1024:.0f}K)")
        print(f"{'='*60}")
        
        # 根据序列长度调整训练配置
        if sl <= 65536:  # 32K-64K
            steps = 1000
            eval_interval = 200
            eval_batches = 5
            chunk_size = 1024
        elif sl <= 262144:  # 256K
            steps = 500
            eval_interval = 100
            eval_batches = 3
            chunk_size = 2048
        else:  # 2M
            steps = 200
            eval_interval = 50
            eval_batches = 2
            chunk_size = 4096
        
        set_seed(SEED)
        mh_cfg = MHDSRA2Config(
            dim=128, heads=4, local_window=64, slot_pe="rope",
            momentum_qkv=False, use_context_film=False, slots=64,
            tau_init=16.0, tau_write_init=16.0,
        )
        model = StreamingHybridModel(VOCAB_SIZE, mh_cfg, chunk_size=chunk_size, max_seq_len=sl+1024)
        acc, mem = train_and_evaluate(
            f"Streaming seq_len={sl:,}", model, generate_niah, 
            steps=steps, seq_len=sl, lr=5e-4, distance=distance, pass_distance=True,
            eval_interval=eval_interval, eval_batches=eval_batches
        )
        results[f"seq_len={sl:,}"] = {"acc": acc, "mem": mem, "steps": steps}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 如果32K失败，跳过后续测试
        if sl == 32768 and acc < 0.1:
            print(f"\n⚠️  32K测试失败 (Acc={acc:.3f})，跳过后续长序列测试")
            break
    
    return results


# ============================================================================
# Phase 4: 真实场景验证
# ============================================================================

def phase4_real_world_test():
    """Phase 4: 真实场景验证"""
    print("\n" + "="*70)
    print("Phase 4: 真实场景验证")
    print("="*70)
    
    results = {}
    
    scenarios = [
        {"name": "文档问答", "gen": generate_document_qa, "seq_len": 4096},
        {"name": "代码理解", "gen": generate_code_understanding, "seq_len": 2048},
        {"name": "信息更新", "gen": generate_info_update, "seq_len": 4096},
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        mh_cfg = MHDSRA2Config(
            dim=128, heads=4, local_window=64, slot_pe="rope",
            momentum_qkv=False, use_context_film=False, slots=64,
            tau_init=16.0, tau_write_init=16.0,
        )
        model = HybridModel(
            VOCAB_SIZE, mh_cfg, 
            st_dim=128, st_layers=1, st_heads=4,
            fusion_type="weighted",
            max_seq_len=scenario['seq_len'] + 1024
        )
        acc, mem = train_and_evaluate(
            f"Hybrid ({scenario['name']})", model, scenario['gen'], 
            steps=1500, seq_len=scenario['seq_len'],
            eval_interval=100, eval_batches=10
        )
        results[scenario['name']] = {"acc": acc, "mem": mem}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="MHDSRA2 综合测试")
    parser.add_argument("--phase", choices=["1", "2", "3", "4", "all"], default="all")
    args = parser.parse_args()
    
    print("MHDSRA2 混合架构优化与全面测试")
    print(f"设备: {DEVICE}")
    print(f"Vocab: {VOCAB_SIZE}")
    
    all_results = {}
    
    if args.phase in ["1", "all"]:
        all_results["Phase 1: 混合架构优化"] = phase1_architecture_optimization()
    
    if args.phase in ["2", "all"]:
        all_results["Phase 2: 消融测试"] = phase2_ablation_test()
    
    if args.phase in ["3", "all"]:
        all_results["Phase 3: 2M 序列测试"] = phase3_streaming_test()
    
    if args.phase in ["4", "all"]:
        all_results["Phase 4: 真实场景验证"] = phase4_real_world_test()
    
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    
    for phase_name, results in all_results.items():
        print(f"\n{phase_name}:")
        for test_name, metrics in results.items():
            if "acc" in metrics:
                print(f"  {test_name:<35}: Acc={metrics['acc']:.3f}, Memory={metrics.get('mem', 0):.1f} MB")
            else:
                print(f"  {test_name:<35}: {metrics}")


if __name__ == "__main__":
    main()
