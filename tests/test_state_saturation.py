from pathlib import Path

import torch

from dsra_layer import DSRA_Chunk_Layer
from src.dsra.report_utils import ensure_reports_dir

def run_saturation_case(K, dim, seq_len, chunk_size, decay_lambda, title):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Running Saturation Test: {title} ---")
    print(f"K={K}, dim={dim}, seq_len={seq_len}, lambda={decay_lambda}")
    
    # We use a very small K to force rapid saturation
    layer = DSRA_Chunk_Layer(dim, K=K, kr=max(1, K//2), decay_lambda=decay_lambda).to(device)
    layer.eval()
    
    B = 1
    # Generate completely random noise sequence (highly novel at every step)
    x = torch.randn(B, seq_len, dim).to(device)
    
    S_prev = None
    bypass_kv = None
    S_time_prev = None
    v_orth_norms = []
    
    chunk_idx = 0
    with torch.no_grad():
        for i in range(0, seq_len, chunk_size):
            chunk = x[:, i:i+chunk_size, :]
            out_chunk, S_next, next_bypass_kv, S_time_next = layer(
                chunk, S_prev=S_prev, bypass_kv=bypass_kv, S_time_prev=S_time_prev, chunk_idx=chunk_idx
            )
            
            # Record the norm of the orthogonal projection
            # If it drops to 0, it means the state space is saturated and no new info can be written
            norm = layer.last_V_orth.norm(dim=-1).mean().item()
            v_orth_norms.append(norm)
            
            S_prev = S_next
            bypass_kv = next_bypass_kv
            S_time_prev = S_time_next
            chunk_idx += 1
            
    print(f"Final V_orth norm: {v_orth_norms[-1]:.6f}")
    return v_orth_norms

def plot_results(results_dict):
    try:
        import matplotlib.pyplot as plt
        reports_dir = ensure_reports_dir(Path(__file__).resolve().parents[1])
        output_path = reports_dir / "saturation_test_results.png"
        plt.figure(figsize=(10, 6))
        for title, norms in results_dict.items():
            plt.plot(norms, label=title)
        
        plt.title('Orthogonal Projection Norm over Time (Chunks)')
        plt.xlabel('Chunk Index')
        plt.ylabel('Mean L2 Norm of V_orth')
        plt.yscale('log') # Log scale to better see the drop to near-zero
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"\nPlot saved to {output_path}")
    except ImportError:
        print("\nmatplotlib not installed, skipping plot generation.")
        print("Data Summary:")
        for title, norms in results_dict.items():
            print(f"  {title}: start={norms[0]:.4f}, end={norms[-1]:.4f}, min={min(norms):.4f}")

def run_saturation_test():
    dim = 64
    K = 4 # Tiny capacity to force quick saturation
    seq_len = 16384 # Extremely long relative to K
    chunk_size = 64
    
    results = {}
    
    # 1. No Decay (Should saturate and drop to 0)
    results['No Decay (lambda=0)'] = run_saturation_case(
        K, dim, seq_len, chunk_size, decay_lambda=0.0, title="No Decay"
    )
    
    # 2. Mild Decay (Should maintain a steady state > 0)
    results['Mild Decay (lambda=0.05)'] = run_saturation_case(
        K, dim, seq_len, chunk_size, decay_lambda=0.05, title="Mild Decay"
    )
    
    # 3. Strong Decay (Should maintain a higher steady state)
    results['Strong Decay (lambda=0.2)'] = run_saturation_case(
        K, dim, seq_len, chunk_size, decay_lambda=0.2, title="Strong Decay"
    )
    
    plot_results(results)

if __name__ == '__main__':
    run_saturation_test()
