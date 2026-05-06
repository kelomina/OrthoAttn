"""Tiny LLaMA-style LM training with Standard Causal Attention."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tiny_llama_shared import (
    LMConfig,
    CharTokenizer,
    download_wikitext2,
    load_text,
    create_dataloader,
    resolve_device,
)


class RoPE(nn.Module):
    """Rotary Position Embedding for causal attention."""

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_len = max_len

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x. x: [B, H, T, D]"""
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    cos = cos[: x.size(2), :].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.size(2), :].unsqueeze(0).unsqueeze(0)
    x_rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + x_rot * sin


class CausalSelfAttention(nn.Module):
    """Standard causal multi-head self-attention with RoPE."""

    def __init__(self, dim: int, heads: int, max_len: int = 4096):
        super().__init__()
        self.heads = heads
        self.d_head = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.d_head, max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, T, d]

        cos = self.rope(T, x.device).cos().to(dtype=x.dtype)
        sin = self.rope(T, x.device).sin().to(dtype=x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, T, D))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_hidden: int | None = None):
        super().__init__()
        ffn_hidden = ffn_hidden or 4 * dim
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class StandardAttentionLM(nn.Module):
    """Tiny LLaMA with standard causal attention."""

    def __init__(self, vocab_size: int, dim: int = 256, heads: int = 4,
                 num_layers: int = 6, max_len: int = 4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(self.ln_final(h))


def train_standard_lm(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    config: dict,
    device: torch.device,
) -> float:
    """Train Standard Attention LM and return final perplexity."""
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_steps"])
    criterion = nn.CrossEntropyLoss()
    model.train()

    total_steps = 0
    best_ppl = float("inf")
    step_time = 0.0

    while total_steps < config["max_steps"]:
        for batch_x, batch_y in train_loader:
            t0 = time.time()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step_time = time.time() - t0

            total_steps += 1

            if total_steps % config["eval_interval"] == 0:
                ppl = math.exp(loss.item())
                best_ppl = min(best_ppl, ppl)
                print(
                    f"[Standard] Step {total_steps:5d} | Loss: {loss.item():.4f} | "
                    f"PPL: {ppl:.2f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Step: {step_time*1000:.0f}ms"
                )

            if total_steps >= config["max_steps"]:
                break

    print(f"\n[Standard] Training complete. Best PPL: {best_ppl:.2f}")
    return best_ppl


def main_standard(config: dict | None = None) -> float:
    """Entry point: run Standard LM training and return final PPL."""
    cfg = dict(LMConfig)
    if config:
        cfg.update(config)

    device = resolve_device(cfg["device"])
    tokenizer = CharTokenizer()
    print(f"[Standard] Device: {device}, Vocab: {tokenizer.vocab_size}")

    data_path = download_wikitext2(cfg["data_dir"])
    text = load_text(data_path, max_chars=2_000_000)
    train_loader = create_dataloader(
        text, tokenizer, cfg["seq_len"], cfg["batch_size"], shuffle=True,
    )
    print(f"[Standard] Data: {len(train_loader.dataset)} sequences")

    model = StandardAttentionLM(
        vocab_size=tokenizer.vocab_size,
        dim=cfg["dim"],
        heads=cfg["heads"],
        num_layers=cfg["num_layers"],
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Standard] Model: {total_params:,} parameters")

    final_ppl = train_standard_lm(model, train_loader, cfg, device)
    return final_ppl


if __name__ == "__main__":
    main_standard()
