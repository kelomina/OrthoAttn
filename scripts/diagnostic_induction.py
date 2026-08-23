import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dsra.domain.mqar import MQARConfig, generate_mqar_batch
from scripts.benchmark_mqar import evaluate_mqar


class CleanInductionTransformer(nn.Module):
    """Transformer baseline for MQAR without distance decay on content matching."""
    def __init__(self, vocab_size=256, dim=128, heads=4, num_layers=2, max_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        
        # We test standard MultiheadAttention
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=4 * dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_emb(pos)
        
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        return self.out_proj(self.ln_final(h))


class RoPECustomTransformer(nn.Module):
    """Transformer with Layer 1 local RoPE / pos and Layer 2 pure content attention."""
    def __init__(self, vocab_size=256, dim=128, heads=4, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.heads = heads
        self.d_head = dim // heads
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Local 1D conv or short embedding to bind adjacent (K, V)
        self.kv_bind = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=4 * dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        h = self.embedding(x)
        # 1D causal / local conv
        h = h + self.kv_bind(h.transpose(1, 2)).transpose(1, 2)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        return self.out_proj(self.ln_final(h))


def test_model(model_cls, name, seq_len=512, num_kv=4, epochs=500, lr=1e-3, bs=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = MQARConfig(vocab_size=256, seq_len=seq_len, num_kv_pairs=num_kv, num_queries=num_kv)
    model = model_cls().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    print(f"\n--- Testing {name} on L={seq_len}, K={num_kv} ---")
    for step in range(epochs):
        model.train()
        X, Y, qpos, targets = generate_mqar_batch(bs, cfg, device=device, seed=70000 + step)
        opt.zero_grad()
        logits = model(X)
        loss = crit(logits.view(-1, cfg.vocab_size), Y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 50 == 0:
            metrics = evaluate_mqar(model, cfg, device, eval_batches=5, batch_size=8)
            print(f"Step {step+1:3d}/{epochs} | train_loss={loss.item():.4f} | eval_loss={metrics['loss']:.4f} | eval_acc={metrics['accuracy']*100:5.1f}%")
            if metrics["accuracy"] >= 0.90:
                print(f"Success! {name} reached >= 90% ({metrics['accuracy']*100:.1f}%) at step {step+1}")
                break


if __name__ == "__main__":
    test_model(RoPECustomTransformer, "Conv-Bound Transformer", seq_len=512, num_kv=4, epochs=400, lr=2e-3, bs=32)
    test_model(CleanInductionTransformer, "Clean APE Transformer", seq_len=512, num_kv=4, epochs=400, lr=2e-3, bs=32)
