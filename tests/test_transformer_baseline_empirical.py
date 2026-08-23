"""Empirical convergence verification test for Standard Causal Transformer on MQAR."""

import pytest
import torch
import torch.nn.functional as F
from src.dsra.domain.mqar import MQARConfig, generate_mqar_batch
from scripts.benchmark_mqar import (
    StandardCausalTransformer,
    evaluate_mqar,
    get_cosine_warmup_scheduler,
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_transformer_convergence_short(device):
    """Verify that Standard Causal Transformer can converge on a simple MQAR setup."""
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    cfg = MQARConfig(vocab_size=64, seq_len=64, num_kv_pairs=1, num_queries=1)
    model = StandardCausalTransformer(vocab_size=64, dim=64, heads=2, num_layers=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)

    for step in range(150):
        model.train()
        X, Y, qpos, targets = generate_mqar_batch(32, cfg, device=device, seed=1000 + step)
        opt.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), Y.view(-1), ignore_index=0)
        loss.backward()
        opt.step()

    metrics = evaluate_mqar(model, cfg, device, eval_batches=5, batch_size=8)
    print(f"\nShort MQAR eval: acc={metrics['accuracy']*100:.1f}%, loss={metrics['loss']:.4f}")
    assert metrics["accuracy"] >= 0.90


