
"""Diagnostic test for write protection feature to check for potential bugs."""
import torch

from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2

torch.manual_seed(42)
device = torch.device('cpu')

# Configuration
cfg = MHDSRA2Config(
    dim=32,
    heads=2,
    slots=4,
    read_topk=2,
    write_topk=2,
    local_window=4,
    use_local=True,
    use_retrieval=True,
    detach_state=False,
    write_protection=3,  # 3-token write protection
)

layer = MultiHeadDSRA2(cfg).to(device)

print("=== Testing Write Protection ===")

# First forward pass - write to some slots
batch = 1
x1 = torch.randn(batch, 2, cfg.dim).to(device)
y1, state1 = layer(x1)
print(f"State after first pass:")
print(f"  Position: {state1.position}")
print(f"  Protected until: {state1.protected_until}")
print(f"  Slot k: {state1.slot_k[0, 0, :, 0]}")

# Save first state slot_k for later comparison
slot_k1 = state1.slot_k.clone()
slot_v1 = state1.slot_v.clone()

# Second forward pass - should NOT overwrite recently written slots
x2 = torch.randn(batch, 2, cfg.dim).to(device)
y2, state2 = layer(x2, state=state1)
print(f"\nState after second pass:")
print(f"  Position: {state2.position}")
print(f"  Protected until: {state2.protected_until}")
print(f"  Slot k: {state2.slot_k[0, 0, :, 0]}")

# Check which slots changed
slot_k2 = state2.slot_k.clone()
slot_v2 = state2.slot_v.clone()

# Check for absolute changes
changed_k = torch.any(torch.abs(slot_k1 - slot_k2) > 1e-6, dim=(1, 2, 3))
changed_v = torch.any(torch.abs(slot_v1 - slot_v2) > 1e-6, dim=(1, 2, 3))
print(f"\nSlots changed between first and second pass:")
print(f"  K changed: {changed_k}")
print(f"  V changed: {changed_v}")

# Now let's examine the write protection logic step by step
print("\n=== Detailed Write Protection Debug ===")
# Let's manually trace through what should happen
print(f"Write protection window: {cfg.write_protection}")
print(f"Initial position: 0")
print(f"After first pass (+2 tokens): position={state1.position}")
print(f"Protected until after first pass (position + len + write_protection): {state1.position}+2+{cfg.write_protection} = {state1.position+2+cfg.write_protection}")
print(f"Second pass starts at position={state1.position}, which is < protected_until")
print(f"Therefore, recently written slots should NOT change in second pass")

# Third forward pass - should now be outside protection window, can overwrite
x3 = torch.randn(batch, 2, cfg.dim).to(device)
y3, state3 = layer(x3, state=state2)
print(f"\nState after third pass:")
print(f"  Position: {state3.position}")
print(f"  Protected until: {state3.protected_until}")

# Check for changes between second and third
slot_k3 = state3.slot_k.clone()
changed_k23 = torch.any(torch.abs(slot_k2 - slot_k3) > 1e-6, dim=(1, 2, 3))
print(f"\nSlots changed between second and third pass: {changed_k23}")
