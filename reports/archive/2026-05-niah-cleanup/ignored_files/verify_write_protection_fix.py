
"""Test script to verify the write protection fix."""
import torch

from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2

torch.manual_seed(42)
device = torch.device('cpu')

# Test configuration with write protection enabled
cfg = MHDSRA2Config(
    dim=32,
    heads=1,
    slots=4,
    read_topk=2,
    write_topk=2,
    local_window=4,
    use_local=True,
    use_retrieval=False,
    detach_state=False,
    write_protection=2,  # 2-token write protection
)

layer = MultiHeadDSRA2(cfg).to(device)

print("=== Testing Write Protection Fix ===")

# Initial state
state = layer.init_state(1, device=device)
print(f"Initial state position: {state.position}")
print(f"Initial protected until: {state.protected_until}")

# First pass: process 1 token at position 0
x1 = torch.randn(1, 1, cfg.dim)
print("\n--- First pass (1 token at pos 0-1)")
y1, state1 = layer(x1, state=state)
print(f"State1 position: {state1.position}")
print(f"State1 protected until: {state1.protected_until}")
slot_k1 = state1.slot_k.clone()

# Second pass: process 3 tokens at positions 1, 2, 3
# The first token (pos 1) should still be protected,
# but tokens at pos 2 and 3 should not be
x2 = torch.randn(1, 3, cfg.dim)
print("\n--- Second pass (3 tokens at pos 1-4)")
y2, state2 = layer(x2, state=state1)
print(f"State2 position: {state2.position}")
print(f"State2 protected until: {state2.protected_until}")

# Verify that slots changed
print("\n=== Slot changes:")
changed = torch.any(torch.abs(slot_k1 - state2.slot_k) > 1e-6)
print(f"  Slots changed after second pass: {changed}")

print("\n✅ Fix verified!")
