
import torch
from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2


def test_write_protection():
    # Configuration with write protection
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
        write_protection=3,  # Protect slots for 3 positions
    )
    layer = MultiHeadDSRA2(cfg)

    # Initial state
    state = layer.init_state(batch_size=1, device='cpu', dtype=torch.float32)
    print("Initial state position:", state.position)
    print("Initial protected_until:", state.protected_until)

    # First chunk: write to some slots
    x1 = torch.randn(1, 2, 32)  # batch=1, seq_len=2, dim=32
    y1, state1 = layer(x1, state=state)

    print("\nAfter first chunk:")
    print("Position:", state1.position)
    print("Protected_until:", state1.protected_until)

    # Check which slots were written
    wrote_mask1 = (state1.usage > state.usage).cpu().numpy()
    print("Slots written in first chunk:", wrote_mask1)

    # Second chunk: within protection window
    x2 = torch.randn(1, 2, 32)
    y2, state2 = layer(x2, state=state1)

    print("\nAfter second chunk:")
    print("Position:", state2.position)
    print("Protected_until:", state2.protected_until)

    # Check that previously written slots didn't change
    slot_k1 = state1.slot_k.cpu().numpy()
    slot_k2 = state2.slot_k.cpu().numpy()
    slot_changed = ~(abs(slot_k1 - slot_k2) < 1e-6).all(axis=-1)
    print("Slots changed in second chunk:", slot_changed)

    # Verify slots written in first chunk were not overwritten
    protected_slots = wrote_mask1 & slot_changed
    print("Protected slots changed (should be empty):", protected_slots)

    # Ensure protected slots were not changed
    assert not protected_slots.any(), "Protected slots were overwritten!"


if __name__ == "__main__":
    test_write_protection()
    print("\nTest passed!")
