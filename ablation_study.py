import torch
import torch.nn as nn
import torch.optim as optim
from toy_task_associative_recall import generate_associative_recall_data, train_step
from dsra_model import MultiLayerDSRAModel


def run_ablation(name, model, device, epochs=3000, batch_size=16, seq_len=512, vocab_size=100):
    print(f"\n--- Starting Ablation Study: {name} ---")
    # 增加训练轮次以确保模型充分收敛，特别是消融实验需要更长的训练来观察各组件的真实影响
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_acc = 0.0

    model.train()
    for epoch in range(epochs):
        X, Y = generate_associative_recall_data(batch_size, seq_len, vocab_size, num_pairs=10)
        X, Y = X.to(device), Y.to(device)

        loss_val, acc = train_step(model, X, Y, optimizer, criterion)
        best_acc = max(best_acc, acc)

        if epoch % 50 == 0:
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
    
    ablations = [
        ("Full DSRA (NoPE)", {"use_orthogonal_update": True, "use_bypass": True, "pe_mode": "none"}),
        ("No Orthogonal Update", {"use_orthogonal_update": False, "use_bypass": True, "pe_mode": "none"}),
        ("No Bypass", {"use_orthogonal_update": True, "use_bypass": False, "pe_mode": "none"}),
        ("Full DSRA (RoPE)", {"use_orthogonal_update": True, "use_bypass": True, "pe_mode": "rope"}),
        ("Full DSRA (ALiBi)", {"use_orthogonal_update": True, "use_bypass": True, "pe_mode": "alibi"}),
        ("Full DSRA (Timestamps)", {"use_orthogonal_update": True, "use_bypass": True, "pe_mode": "timestamps"}),
    ]

    results = {}
    for name, config in ablations:
        model = MultiLayerDSRAModel(
            vocab_size, dim, num_layers, K, kr, chunk_size,
            use_orthogonal_update=config["use_orthogonal_update"],
            use_bypass=config["use_bypass"],
            pe_mode=config["pe_mode"]
        ).to(device)
        results[name] = run_ablation(name, model, device, epochs=3000, seq_len=seq_len, vocab_size=vocab_size)

    print("\n=== Ablation Study Results ===")
    for name, _ in ablations:
        print(f"{name} Accuracy: {results[name]*100:.1f}%")

if __name__ == '__main__':
    main()
