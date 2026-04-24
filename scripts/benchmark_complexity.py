import torch
import time
import math
from src.dsra.dsra_layer import DSRA_Chunk_Layer
import torch.nn.functional as F

def benchmark_dsra(seq_lengths, dim=128, K=128, kr=16, chunk_size=256, batch_size=4, skip_attn_after_inf=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on device: {device}")

    model = DSRA_Chunk_Layer(dim, K=K, kr=kr).to(device)
    model.eval()

    results = []
    skip_attn = False

    for seq_len in seq_lengths:
        try:
            x = torch.randn(batch_size, seq_len, dim).to(device)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            start_time = time.perf_counter()

            S_prev = None
            bypass_kv = None
            S_time_prev = None

            with torch.no_grad():
                chunk_idx = 0
                for i in range(0, seq_len, chunk_size):
                    chunk = x[:, i:i+chunk_size, :]
                    out_chunk, S_next, next_bypass_kv, S_time_next = model(
                        chunk, S_prev=S_prev, bypass_kv=bypass_kv, S_time_prev=S_time_prev, chunk_idx=chunk_idx
                    )
                    S_prev = S_next
                    bypass_kv = next_bypass_kv
                    S_time_prev = S_time_next
                    chunk_idx += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                peak_mem = 0.0

            end_time = time.perf_counter()
            fwd_time = (end_time - start_time) * 1000

            attn_time = float('inf')
            if not skip_attn:
                try:
                    start_attn = time.perf_counter()
                    with torch.no_grad():
                        Q = torch.randn(batch_size, seq_len, dim).to(device)
                        K_attn = torch.randn(batch_size, seq_len, dim).to(device)
                        V_attn = torch.randn(batch_size, seq_len, dim).to(device)
                        _ = F.scaled_dot_product_attention(Q, K_attn, V_attn, is_causal=True)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_attn = time.perf_counter()
                    attn_time = (end_attn - start_attn) * 1000
                    attn_measured = True
                    del Q, K_attn, V_attn
                except torch.cuda.OutOfMemoryError:
                    attn_time = float('inf')
                except Exception as e:
                    attn_time = float('inf')

            if skip_attn_after_inf and not skip_attn and attn_time == float('inf'):
                skip_attn = True
                print(f"Std Attn reached inf at seq_len={seq_len}, skipping for longer sequences...")

            results.append({
                'seq_len': seq_len,
                'fwd_time_ms': fwd_time,
                'peak_mem_mb': peak_mem,
                'attn_time_ms': attn_time
            })

            if skip_attn:
                print(f"SeqLen: {seq_len:<6} | DSRA Time: {fwd_time:8.2f} ms | Std Attn Time:   SKIP    | Peak Mem: {peak_mem:6.2f} MB")
            else:
                print(f"SeqLen: {seq_len:<6} | DSRA Time: {fwd_time:8.2f} ms | Std Attn Time: {attn_time:8.2f} ms | Peak Mem: {peak_mem:6.2f} MB")

            del x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"SeqLen: {seq_len:<6} | DSRA Time:      OOM | Std Attn Time:      OOM | Peak Mem:   OOM MB")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results

def run_benchmark():
    print("--- DSRA Complexity Benchmark ---")
    print("Testing O(N) linear time and decoupled memory footprint.")
    lengths = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]
    benchmark_dsra(lengths, skip_attn_after_inf=True)

if __name__ == '__main__':
    run_benchmark()