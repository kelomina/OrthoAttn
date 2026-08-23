import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dsra.domain.mqar import MQARConfig, generate_mqar_batch
from scripts.benchmark_mqar import (
    StandardCausalTransformer,
    evaluate_mqar,
    get_cosine_warmup_scheduler,
)


def test_config(
    name: str,
    seq_len: int = 512,
    num_kv: int = 4,
    dim: int = 128,
    heads: int = 4,
    num_layers: int = 2,
    lr: float = 1e-3,
    epochs: int = 500,
    batch_size: int = 16,
    warmup: int = 50,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = MQARConfig(vocab_size=256, seq_len=seq_len, num_kv_pairs=num_kv, num_queries=num_kv)
    model = StandardCausalTransformer(
        vocab_size=256,
        dim=dim,
        heads=heads,
        num_layers=num_layers,
        ffn_hidden=4 * dim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.98))
    sched = get_cosine_warmup_scheduler(opt, warmup_steps=warmup, total_steps=epochs)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    print(f"\n--- Testing [{name}]: L={seq_len}, K={num_kv}, dim={dim}, layers={num_layers}, lr={lr}, epochs={epochs}, bs={batch_size} ---")
    t0 = time.time()
    best_acc = 0.0

    for step in range(epochs):
        model.train()
        X, Y, qpos, targets = generate_mqar_batch(batch_size, cfg, device=device, seed=42000 + step)
        opt.zero_grad()
        logits = model(X)
        loss = crit(logits.view(-1, cfg.vocab_size), Y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % 50 == 0 or step == epochs - 1:
            eval_metrics = evaluate_mqar(model, cfg, device, eval_batches=5, batch_size=4)
            acc = eval_metrics["accuracy"]
            if acc > best_acc:
                best_acc = acc
            print(f"Step {step+1:4d}/{epochs} | train_loss={loss.item():.4f} | eval_loss={eval_metrics['loss']:.4f} | eval_acc={acc*100:5.1f}% (best={best_acc*100:5.1f}%) | time={time.time()-t0:.1f}s")
            if acc >= 0.999 and step >= 100:
                print(f"Early converged at step {step+1}!")
                break

    return best_acc


if __name__ == "__main__":
    # Test A: L=512, K=4 with bs=64, lr=3e-3, 1000 steps
    test_config("L512_K4_bs64_lr3e-3", seq_len=512, num_kv=4, dim=128, lr=3e-3, epochs=1200, batch_size=64, warmup=100)
    # Test B: L=128, K=2 with bs=64, lr=3e-3, 600 steps
    test_config("L128_K2_bs64_lr3e-3", seq_len=128, num_kv=2, dim=128, lr=3e-3, epochs=600, batch_size=64, warmup=50)
