import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dsra.domain.mqar import MQARConfig, generate_mqar_batch
from scripts.benchmark_mqar import (
    StandardCausalTransformer,
    evaluate_mqar,
    get_cosine_warmup_scheduler,
)


def test_depth_and_schedule(num_layers=4, dim=128, heads=4, lr=3e-3, epochs=1500, bs=64, seq_len=512, num_kv=4):
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
    sched = get_cosine_warmup_scheduler(opt, warmup_steps=100, total_steps=epochs)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    print(f"\n--- Testing Layers={num_layers}, Dim={dim}, LR={lr}, BS={bs}, Epochs={epochs}, L={seq_len}, K={num_kv} ---")
    best_acc = 0.0
    for step in range(epochs):
        model.train()
        X, Y, qpos, targets = generate_mqar_batch(bs, cfg, device=device, seed=60000 + step)
        opt.zero_grad()
        logits = model(X)
        loss = crit(logits.view(-1, cfg.vocab_size), Y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % 100 == 0 or step == epochs - 1:
            metrics = evaluate_mqar(model, cfg, device, eval_batches=5, batch_size=8)
            acc = metrics["accuracy"]
            if acc > best_acc:
                best_acc = acc
            print(f"Step {step+1:4d}/{epochs} | train_loss={loss.item():.4f} | eval_loss={metrics['loss']:.4f} | eval_acc={acc*100:5.1f}% (best={best_acc*100:5.1f}%)")
            if acc >= 0.95:
                print(f"Goal >= 90.0% reached at step {step+1} ({acc*100:.1f}%)!")
                break


if __name__ == "__main__":
    test_depth_and_schedule(num_layers=4, dim=128, heads=4, lr=3e-3, epochs=1200, bs=64, seq_len=512, num_kv=4)
    test_depth_and_schedule(num_layers=2, dim=128, heads=4, lr=4e-3, epochs=1500, bs=64, seq_len=512, num_kv=4)
