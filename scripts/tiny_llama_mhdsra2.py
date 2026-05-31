"""Tiny LLaMA-style LM training with MHDSRA2 attention."""
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
    PAD_ID,
    load_text,
    create_dataloader,
    create_eval_loader,
    load_wikitext2_splits,
    split_train_validation_text,
    resolve_device,
)


from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2


class MHDSRA2WithFFN(nn.Module):
    """MHDSRA2 layers interleaved with FFN for fair LM comparison.

    Architecture per layer:
      x → LN → MHDSRA2(chunked, state) → residual + → LN → FFN → residual +
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        heads: int = 4,
        num_layers: int = 6,
        slots: int = 64,
        chunk_size: int = 128,
        mhdsra2_config_override: dict | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.chunk_size = chunk_size

        self.embedding = nn.Embedding(vocab_size, dim)
        self.final_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

        # Build MHDSRA2 config
        cfg = MHDSRA2Config(
            dim=dim,
            heads=heads,
            slots=slots,
            read_topk=max(1, min(slots // 8, slots)),
            write_topk=max(1, min(slots // 16, slots)),
            local_window=chunk_size * 4,  # allow longer local context
            use_local=True,
            use_retrieval=False,
            detach_state=False,
            slot_pe="none",
        )
        if mhdsra2_config_override:
            for k, v in mhdsra2_config_override.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

        # MHDSRA2 layers
        self.mhdsra2_layers = nn.ModuleList([
            MultiHeadDSRA2(cfg) for _ in range(num_layers)
        ])
        # Pre-attention norms
        self.pre_attn_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
        # FFN layers
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 4 * dim),
                nn.GELU(),
                nn.Linear(4 * dim, dim),
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        hidden = self.embedding(x)
        state_list = [None] * self.num_layers
        out_list = []

        for start in range(0, T, self.chunk_size):
            chunk = hidden[:, start: start + self.chunk_size, :]
            for layer_idx in range(self.num_layers):
                residual = chunk
                chunk = self.pre_attn_norms[layer_idx](chunk)
                chunk, state_list[layer_idx] = self.mhdsra2_layers[layer_idx](
                    chunk, state=state_list[layer_idx],
                )
                chunk = residual + chunk
                chunk = chunk + self.ffns[layer_idx](chunk)
            out_list.append(chunk)

        out = torch.cat(out_list, dim=1)
        out = self.final_norm(out)
        return self.out_proj(out)


def build_mhdsra2_lm(
    vocab_size: int,
    dim: int = 256,
    heads: int = 4,
    num_layers: int = 6,
    slots: int = 64,
    chunk_size: int = 128,
    mhdsra2_config_override: dict | None = None,
) -> MHDSRA2WithFFN:
    """Build a causal language model backed by MHDSRA2 layers + FFN."""
    return MHDSRA2WithFFN(
        vocab_size=vocab_size,
        dim=dim,
        heads=heads,
        num_layers=num_layers,
        slots=slots,
        chunk_size=chunk_size,
        mhdsra2_config_override=mhdsra2_config_override,
    )


def train_mhdsra2_lm(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    config: dict,
    device: torch.device,
) -> nn.Module:
    """Train MHDSRA2 LM and return the trained model."""
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_steps"])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    model.train()

    total_steps = 0
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
                print(
                    f"[MHDSRA2] Step {total_steps:5d} | Loss: {loss.item():.4f} | "
                    f"Train Batch PPL: {ppl:.2f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Step: {step_time*1000:.0f}ms"
                )

            if total_steps >= config["max_steps"]:
                break

    print("\n[MHDSRA2] Training complete.")
    return model


def evaluate_ppl(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Evaluate perplexity on a dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
            valid_tokens = (batch_y != PAD_ID).sum().item()
            total_loss += loss.item()
            total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError("Cannot evaluate PPL because the evaluation loader has no non-PAD tokens.")
    return math.exp(total_loss / total_tokens)


def main_mhdsra2(config: dict | None = None) -> float:
    """Entry point: run MHDSRA2 LM training and return final PPL."""
    cfg = dict(LMConfig)
    if config:
        cfg.update(config)

    device = resolve_device(cfg["device"])
    tokenizer = CharTokenizer()
    print(f"[MHDSRA2] Device: {device}, Vocab: {tokenizer.vocab_size}")

    # Data
    data_paths = load_wikitext2_splits(cfg["data_dir"])
    text = load_text(data_paths["train"], max_chars=2_200_000)
    if data_paths["valid"] is None:
        text, valid_text = split_train_validation_text(text, validation_chars=200_000)
    else:
        text = text[:2_000_000]
        valid_text = load_text(data_paths["valid"], max_chars=200_000)
    train_loader = create_dataloader(
        text, tokenizer, cfg["seq_len"], cfg["batch_size"], shuffle=True,
    )
    valid_loader = create_eval_loader(
        valid_text, tokenizer, cfg["seq_len"], cfg["batch_size"],
    )
    print(f"[MHDSRA2] Data: {len(train_loader.dataset)} sequences, seq_len={cfg['seq_len']}")

    # Model
    model = build_mhdsra2_lm(
        vocab_size=tokenizer.vocab_size,
        dim=cfg["dim"],
        heads=cfg["heads"],
        num_layers=cfg["num_layers"],
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MHDSRA2] Model: {total_params:,} parameters")

    # Train
    train_mhdsra2_lm(model, train_loader, cfg, device)
    validation_ppl = evaluate_ppl(model, valid_loader, device)
    print(f"[MHDSRA2] Validation PPL: {validation_ppl:.2f}")
    return validation_ppl


if __name__ == "__main__":
    main_mhdsra2()
