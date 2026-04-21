import torch
import torch.nn as nn
import torch.optim as optim
from toy_task_associative_recall import generate_associative_recall_data, train_step
from dsra_model import MultiLayerDSRAModel

def run_ablation(name, model, device, epochs=500, batch_size=16, seq_len=512, vocab_size=100):
    print(f"\n--- Starting Ablation Study: {name} ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_acc = 0.0
    
    model.train()
    for epoch in range(epochs):
        X, Y = generate_associative_recall_data(batch_size, seq_len, vocab_size, num_pairs=10)
        X, Y = X.to(device), Y.to(device)
        
        loss_val, acc = train_step(model, X, Y, optimizer, criterion)
        
        if epoch % 50 == 0:
            best_acc = max(best_acc, acc)
            print(f"[{name}] Epoch {epoch:4d} | Loss: {loss_val:.4f} | Accuracy: {acc*100:.1f}%")
            
            if acc == 1.0 and loss_val < 0.05:
                print(f"[{name}] Solved at epoch {epoch}!")
                break
                
    return best_acc

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    vocab_size = 100
    dim = 64
    seq_len = 512
    chunk_size = 128
    num_layers = 2
    K = 64
    kr = 8
    
    # 1. Full DSRA (NoPE)
    model_full = MultiLayerDSRAModel(
        vocab_size, dim, num_layers, K, kr, chunk_size,
        use_orthogonal_update=True, use_bypass=True, pe_mode='none'
    ).to(device)
    
    # 2. No Orthogonal Update
    model_no_ortho = MultiLayerDSRAModel(
        vocab_size, dim, num_layers, K, kr, chunk_size,
        use_orthogonal_update=False, use_bypass=True, pe_mode='none'
    ).to(device)
    
    # 3. No Instruction Bypass
    model_no_bypass = MultiLayerDSRAModel(
        vocab_size, dim, num_layers, K, kr, chunk_size,
        use_orthogonal_update=True, use_bypass=False, pe_mode='none'
    ).to(device)
    
    # 4. Full DSRA (RoPE)
    model_rope = MultiLayerDSRAModel(
        vocab_size, dim, num_layers, K, kr, chunk_size,
        use_orthogonal_update=True, use_bypass=True, pe_mode='rope'
    ).to(device)
    
    # 5. Full DSRA (ALiBi)
    model_alibi = MultiLayerDSRAModel(
        vocab_size, dim, num_layers, K, kr, chunk_size,
        use_orthogonal_update=True, use_bypass=True, pe_mode='alibi'
    ).to(device)
    
    # 6. Full DSRA (Timestamps)
    model_timestamps = MultiLayerDSRAModel(
        vocab_size, dim, num_layers, K, kr, chunk_size,
        use_orthogonal_update=True, use_bypass=True, pe_mode='timestamps'
    ).to(device)
    
    acc_full = run_ablation("Full DSRA (NoPE)", model_full, device)
    acc_no_ortho = run_ablation("No Orthogonal Update", model_no_ortho, device)
    acc_no_bypass = run_ablation("No Bypass", model_no_bypass, device)
    acc_rope = run_ablation("Full DSRA (RoPE)", model_rope, device)
    acc_alibi = run_ablation("Full DSRA (ALiBi)", model_alibi, device)
    acc_timestamps = run_ablation("Full DSRA (Timestamps)", model_timestamps, device)
    
    print("\n=== Ablation Study Results ===")
    print(f"Full DSRA (NoPE) Accuracy:         {acc_full*100:.1f}%")
    print(f"No Orthogonal Update Acc:   {acc_no_ortho*100:.1f}%")
    print(f"No Instruction Bypass Acc:  {acc_no_bypass*100:.1f}%")
    print(f"Full DSRA (RoPE) Accuracy:         {acc_rope*100:.1f}%")
    print(f"Full DSRA (ALiBi) Accuracy:        {acc_alibi*100:.1f}%")
    print(f"Full DSRA (Timestamps) Acc:        {acc_timestamps*100:.1f}%")

if __name__ == '__main__':
    main()