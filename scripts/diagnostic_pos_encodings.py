import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dsra.domain.mqar import MQARConfig, generate_mqar_batch
from scripts.benchmark_mqar import evaluate_mqar


class LearnedPosTransformer(nn.Module):
    def __init__(self, vocab_size=256, dim=128, heads=4, num_layers=2, max_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_len, dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_final = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.out_proj(self.ln_final(h))


class NoPosTransformer(nn.Module):
    def __init__(self, vocab_size=256, dim=128, heads=4, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_final = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        h = self.embedding(x)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.out_proj(self.ln_final(h))


def run_experiment(model_cls, name, seq_len=512, num_kv=4, epochs=400, lr=1e-3, bs=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = MQARConfig(vocab_size=256, seq_len=seq_len, num_kv_pairs=num_kv, num_queries=num_kv)
    model = model_cls(vocab_size=256, dim=128, heads=4, num_layers=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    print(f"\n--- Testing {name}: L={seq_len}, K={num_kv}, epochs={epochs}, lr={lr}, bs={bs} ---")
    for step in range(epochs):
        model.train()
        X, Y, qpos, targets = generate_mqar_batch(bs, cfg, device=device, seed=50000 + step)
        opt.zero_grad()
        logits = model(X)
        loss = crit(logits.view(-1, cfg.vocab_size), Y.view(-1))
        loss.backward()
        opt.step()

        if (step + 1) % 50 == 0:
            metrics = evaluate_mqar(model, cfg, device, eval_batches=5, batch_size=8)
            print(f"Step {step+1:3d}/{epochs} | train_loss={loss.item():.4f} | eval_loss={metrics['loss']:.4f} | eval_acc={metrics['accuracy']*100:5.1f}%")


if __name__ == "__main__":
    run_experiment(LearnedPosTransformer, "Learned APE Transformer", seq_len=512, num_kv=4, epochs=400, lr=2e-3, bs=32)
    run_experiment(NoPosTransformer, "No-Pos Transformer", seq_len=512, num_kv=4, epochs=400, lr=2e-3, bs=32)
