# -*- coding: utf-8 -*-
"""门控熵正则回归测试.

中文说明:
- 被测对象 / Targets: `MHDSRA2Config.gate_entropy_weight`、
  `MultiHeadDSRA2._forward_from_projected` 的熵亏空计算、
  `MultiLayerMHDSRA2Model.forward(return_aux=...)` 的辅助量管道
- 调用方 / Called by: `pytest tests/test_gate_entropy_regularization.py`
- 覆盖 / Coverage:
  1) 默认权重 0.0 且 forward 不带 return_aux 时行为与历史签名完全一致；
  2) 非法负权重被 `__post_init__` 拒绝；
  3) return_aux=True 返回 (logits, payload)，gate_entropy_loss 为有限非负标量；
  4) 门控越接近 one-hot，熵亏空越大（均匀 < 中等 < one-hot 单调性）；
  5) 权重为 0 时即使请求 aux 也返回 None。
"""

import math

import pytest
import torch

from src.dsra.dsra_model import MultiLayerMHDSRA2Model
from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config


def _build_model(vocab_size: int = 32, **overrides) -> MultiLayerMHDSRA2Model:
    """构建 CPU 上的最小可运行多层模型."""
    kwargs = dict(
        vocab_size=vocab_size,
        dim=32,
        num_layers=1,
        K=8,
        kr=2,
        chunk_size=16,
        use_retrieval=False,
        mhdsra2_config_override={"detach_state": False, **overrides},
    )
    return MultiLayerMHDSRA2Model(**kwargs)


class _FixedGate(torch.nn.Module):
    """固定 logits 的替身 fuse_gate，用于构造可控门控分布."""

    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

    def forward(self, q: torch.Tensor) -> torch.Tensor:  # noqa: D102
        logits = torch.zeros(q.shape[:-1] + (3,), device=q.device, dtype=q.dtype)
        if self.mode == "onehot":
            logits[..., 0] = 12.0
            logits[..., 1:] = -12.0
        elif self.mode == "peaked":
            logits[..., 0] = 3.0
            logits[..., 1:] = -3.0
        # "uniform" 保持全 0
        return logits


def test_default_config_zero_weight_and_signature_unchanged() -> None:
    """默认 gate_entropy_weight=0；不带 return_aux 的 forward 返回纯 logits 张量."""
    cfg = MHDSRA2Config(dim=32, heads=8, slots=8)
    assert cfg.gate_entropy_weight == 0.0
    model = _build_model()
    x = torch.randint(1, 32, (2, 64))
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 64, 32)


def test_negative_weight_rejected() -> None:
    """负的 gate_entropy_weight 必须在配置校验阶段抛出 ValueError."""
    with pytest.raises(ValueError):
        MHDSRA2Config(dim=32, heads=8, slots=8, gate_entropy_weight=-0.1)


def test_return_aux_payload_finite_nonnegative() -> None:
    """return_aux=True 返回 (logits, payload)，熵亏空为有限非负标量."""
    model = _build_model(gate_entropy_weight=0.5)
    x = torch.randint(1, 32, (2, 48))
    logits, payload = model(x, return_aux=True)
    assert isinstance(logits, torch.Tensor)
    value = payload["gate_entropy_loss"]
    assert isinstance(value, torch.Tensor)
    assert value.dim() == 0
    assert torch.isfinite(value).item()
    assert value.item() >= 0.0


def test_gate_entropy_loss_carries_gradient() -> None:
    """熵亏空必须可微：反向传播能到达 fuse_gate 参数（防止误加 detach 变成纯日志量）."""
    model = _build_model(gate_entropy_weight=1.0)
    x = torch.randint(1, 32, (2, 48))
    _, payload = model(x, return_aux=True)
    reg = payload["gate_entropy_loss"]
    assert reg is not None and reg.requires_grad
    reg.backward()
    grads = [
        layer.fuse_gate.weight.grad
        for layer in model.layers
        if layer.fuse_gate.weight.grad is not None
    ]
    assert grads, "gate_entropy_loss 反向传播未触及任何 fuse_gate 参数"
    assert any(float(g.abs().sum()) > 0.0 for g in grads)


def test_weight_zero_returns_none_in_payload() -> None:
    """权重为 0 时即使请求 aux，gate_entropy_loss 也应为 None（不产生额外损失）."""
    model = _build_model()
    x = torch.randint(1, 32, (2, 48))
    _, payload = model(x, return_aux=True)
    assert payload["gate_entropy_loss"] is None


@pytest.mark.parametrize(
    "mode, min_order",
    [
        ("uniform", "low"),
        ("peaked", "mid"),
        ("onehot", "high"),
    ],
)
def test_entropy_deficit_monotonic_under_gate_peakedness(mode: str, min_order: str) -> None:
    """门控越尖锐（接近 one-hot），熵亏空越大：uniform < peaked < onehot."""
    torch.manual_seed(20260506)
    losses = {}
    for mode in ("uniform", "peaked", "onehot"):
        model = _build_model(gate_entropy_weight=1.0)
        for layer in model.layers:
            layer.fuse_gate = _FixedGate(mode)
        x = torch.randint(1, 32, (2, 48))
        with torch.no_grad():
            _, payload = model(x, return_aux=True)
        losses[mode] = float(payload["gate_entropy_loss"])
    assert losses["onehot"] > losses["peaked"] >= losses["uniform"] - 1e-6
    # 理论上界：亏空 <= ln(3)
    assert all(v <= math.log(3) + 1e-4 for v in losses.values())
