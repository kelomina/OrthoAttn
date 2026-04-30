import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.dsra.dsra_model import MultiLayerMHDSRA2Model
from src.dsra.domain import normalize_model_type
from src.dsra.report_utils import build_capacity_markdown, ensure_reports_dir, write_json, write_markdown

DEFAULT_SEQ_LENGTHS = [
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
]

CAPACITY_TEST_LENGTHS = [
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
]


def is_oom_error(exc):
    message = str(exc).lower()
    return "out of memory" in message or "cuda error" in message and "memory" in message

def generate_haystack_with_needle(batch_size, seq_len, vocab_size, needle_depth_ratio=0.5):
    """
    Generate a long sequence of noise (haystack), with a specific unique key-value pair (needle)
    hidden at a specific relative depth.
    
    Vocabulary map:
    0: PAD
    1: QUERY_TOKEN (tells model to answer)
    2: NEEDLE_KEY
    3: NEEDLE_VALUE
    4 to vocab_size-1: Haystack Noise
    """
    X = torch.zeros(batch_size, seq_len, dtype=torch.long)
    Y = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    needle_k, needle_v = 2, 3
    
    for b in range(batch_size):
        # Fill with random haystack tokens
        for i in range(seq_len):
            X[b, i] = random.randint(4, vocab_size - 1)
            
        # Determine needle position based on depth ratio
        # Ratio 0.0 means at the very beginning, 1.0 means at the very end
        # We ensure it's not placed in the last few tokens where the query goes
        max_pos = seq_len - 5
        pos = int(max_pos * needle_depth_ratio)
        
        # Plant the needle
        X[b, pos] = needle_k
        X[b, pos+1] = needle_v
        
        # The query at the very end
        X[b, -2] = needle_k
        X[b, -1] = 1 # QUERY_TOKEN
        
        # Target for the last position is the needle value
        Y[b, -1] = needle_v
        
    return X, Y

def get_niah_runtime_config(seq_len):
    if seq_len <= 32768:
        return {"batch_size": 4, "epochs": 400, "chunk_size": 256}
    if seq_len <= 131072:
        return {"batch_size": 2, "epochs": 250, "chunk_size": 256}
    if seq_len <= 524288:
        return {"batch_size": 1, "epochs": 120, "chunk_size": 256}
    return {"batch_size": 1, "epochs": 60, "chunk_size": 256}


def extract_query_targets(X, Y, logits):
    B, _ = X.shape
    query_indices = (X == 1).nonzero(as_tuple=True)
    if len(query_indices[0]) != B:
        logits_target = logits[:, -1, :]
        targets = Y[:, -1]
    else:
        logits_target = logits[query_indices[0], query_indices[1], :]
        targets = Y[query_indices[0], query_indices[1]]
    return logits_target, targets


def build_niah_model(
    device,
    vocab_size=100,
    dim=64,
    num_layers=2,
    K=64,
    kr=8,
    chunk_size=256,
    model_type="mhdsra2",
):
    """Build the long-context Needle-In-A-Haystack benchmark model.

    中文说明:
    - 调用方 / Called by: `run_single_niah_test`, `run_single_niah_capacity_test`,
      `scripts.next_round_benchmark_runner.run_next_round_benchmark`
    - 调用对象 / Calls: `MultiLayerDSRAModel`, `MultiLayerMHDSRA2Model`
    - 作用 / Purpose: 统一构造 NIAH 基准模型，使 DSRA 与 MHDSRA2 共享相同训练/评测口径
    - 变量 / Variables:
      `model_type` 支持 `dsra/mhdsra2`, 其余参数为维度、层数、槽位与 chunk 配置
    - 接入 / Integration: 新增长上下文模型时优先扩展本函数，避免分散在多个训练入口
    - 错误处理 / Error handling: 未知 `model_type` 抛出 `ValueError`
    - 关键词 / Keywords:
      niah|build_model|dsra|mhdsra2|factory|benchmark|needle|haystack|long_context|构建
    """
    active_model_type = normalize_model_type(model_type)
    if active_model_type == "mhdsra2":
        return MultiLayerMHDSRA2Model(
            vocab_size, dim, num_layers, K, kr, chunk_size
        ).to(device)
    raise ValueError(f"Unsupported model_type: {model_type}")


def run_single_niah_test(
    seq_len,
    device,
    vocab_size=100,
    dim=64,
    num_layers=2,
    K=64,
    kr=8,
    model_type="mhdsra2",
):
    runtime_cfg = get_niah_runtime_config(seq_len)
    batch_size = runtime_cfg["batch_size"]
    epochs = runtime_cfg["epochs"]
    chunk_size = runtime_cfg["chunk_size"]

    print(
        f"\n--- Running Needle-In-A-Haystack Test ({seq_len} tokens) on {device} "
        f"| model_type={model_type} ---"
    )
    print(
        f"Config | batch_size={batch_size} | epochs={epochs} | chunk_size={chunk_size}"
    )

    model = build_niah_model(
        device=device,
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        K=K,
        kr=kr,
        chunk_size=chunk_size,
        model_type=model_type,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    depths_to_test = [0.1, 0.5, 0.9]
    model.train()
    best_overall_acc = 0.0

    for epoch in range(epochs):
        current_depth = random.choice(depths_to_test)
        X, Y = generate_haystack_with_needle(batch_size, seq_len, vocab_size, current_depth)
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        logits = model(X)

        logits_target, targets = extract_query_targets(X, Y, logits)

        loss = criterion(logits_target, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 20 == 0:
            preds = logits_target.argmax(dim=-1)
            correct = (preds == targets).sum().item()
            acc = correct / batch_size
            best_overall_acc = max(best_overall_acc, acc)
            print(f"Epoch {epoch:3d} | Depth: {current_depth:.1f} | Loss: {loss.item():.4f} | Accuracy: {acc*100:5.1f}%")

            if best_overall_acc == 1.0 and loss.item() < 0.1:
                print(f"\nSUCCESS! The model successfully found the needle in {seq_len} context!")
                break

        del X, Y, logits, logits_target, targets, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nFinal Best Accuracy achieved @ {seq_len}: {best_overall_acc*100:.1f}%")
    return best_overall_acc


def run_single_niah_capacity_test(
    seq_len,
    device,
    mode="train_step",
    vocab_size=100,
    dim=64,
    num_layers=2,
    K=64,
    kr=8,
    chunk_size=256,
    batch_size=1,
):
    model = build_niah_model(
        device=device,
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        K=K,
        kr=kr,
        chunk_size=chunk_size,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) if mode == "train_step" else None
    depth = random.choice([0.1, 0.5, 0.9])

    X, Y = generate_haystack_with_needle(batch_size, seq_len, vocab_size, depth)
    X, Y = X.to(device), Y.to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if mode == "forward_only":
        model.eval()
        with torch.no_grad():
            logits = model(X)
            logits_target, targets = extract_query_targets(X, Y, logits)
            loss = criterion(logits_target, targets)
    elif mode == "train_step":
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        logits_target, targets = extract_query_targets(X, Y, logits)
        loss = criterion(logits_target, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    else:
        raise ValueError(f"Unsupported NIAH capacity mode: {mode}")

    preds = logits_target.argmax(dim=-1)
    acc = (preds == targets).float().mean().item()
    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    del X, Y, logits, logits_target, targets, loss, model
    if optimizer is not None:
        del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"status": "ok", "accuracy": acc, "peak_mem_mb": peak_mem_mb}


def run_niah_capacity_test(seq_lengths=None, mode="train_step"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seq_lengths = seq_lengths or CAPACITY_TEST_LENGTHS
    print(f"--- Running Needle-In-A-Haystack Capacity Test ({mode}) on {device} ---")

    results = {}
    for seq_len in seq_lengths:
        print(f"\n--- Capacity Test | mode={mode} | seq_len={seq_len} ---")
        try:
            result = run_single_niah_capacity_test(seq_len=seq_len, device=device, mode=mode)
            results[seq_len] = result
            print(
                f"PASS | seq_len={seq_len} | accuracy={result['accuracy']*100:5.1f}% | "
                f"peak_mem={result['peak_mem_mb']:.2f} MB"
            )
        except torch.cuda.OutOfMemoryError:
            results[seq_len] = {"status": "oom"}
            print(f"OOM | seq_len={seq_len}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            results[seq_len] = {"status": "oom"}
            print(f"OOM | seq_len={seq_len}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except torch.AcceleratorError as exc:
            if not is_oom_error(exc):
                raise
            results[seq_len] = {"status": "oom"}
            print(f"OOM | seq_len={seq_len}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n=== Needle Capacity Results ({mode}) ===")
    for seq_len in seq_lengths:
        result = results[seq_len]
        if result["status"] != "ok":
            print(f"Context {seq_len:>8} | Status: OOM")
        else:
            print(
                f"Context {seq_len:>8} | Status: PASS | "
                f"Accuracy: {result['accuracy']*100:5.1f}% | Peak Mem: {result['peak_mem_mb']:.2f} MB"
            )

    return results


def save_niah_capacity_reports(forward_results, train_results, reports_dir):
    reports_dir = ensure_reports_dir(reports_dir)
    payload = {
        "forward_only": forward_results,
        "train_step": train_results,
    }
    write_json(reports_dir / "needle_capacity_results.json", payload)
    write_markdown(
        reports_dir / "needle_capacity_results.md",
        build_capacity_markdown(forward_results, train_results),
    )


def run_niah_test(seq_lengths=None, model_type="mhdsra2"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seq_lengths = seq_lengths or DEFAULT_SEQ_LENGTHS
    print(
        f"--- Running Needle-In-A-Haystack Long-Context Sweep on {device} "
        f"| model_type={model_type} ---"
    )

    vocab_size = 100
    dim = 64
    num_layers = 2
    K = 64
    kr = 8

    results = {}
    for seq_len in seq_lengths:
        try:
            best_acc = run_single_niah_test(
                seq_len=seq_len,
                device=device,
                vocab_size=vocab_size,
                dim=dim,
                num_layers=num_layers,
                K=K,
                kr=kr,
                model_type=model_type,
            )
            results[seq_len] = best_acc
        except torch.cuda.OutOfMemoryError:
            results[seq_len] = None
            print(f"\nOOM! Skipping Needle-In-A-Haystack at seq_len={seq_len}.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            results[seq_len] = None
            print(f"\nOOM! Skipping Needle-In-A-Haystack at seq_len={seq_len}.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except torch.AcceleratorError as exc:
            if not is_oom_error(exc):
                raise
            results[seq_len] = None
            print(f"\nOOM! Skipping Needle-In-A-Haystack at seq_len={seq_len}.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n=== Needle-In-A-Haystack Sweep Results ===")
    for seq_len in seq_lengths:
        best_acc = results[seq_len]
        if best_acc is None:
            print(f"Context {seq_len:>8} | Best Accuracy: OOM")
        else:
            print(f"Context {seq_len:>8} | Best Accuracy: {best_acc*100:5.1f}%")

    return results

if __name__ == '__main__':
    run_niah_test()
