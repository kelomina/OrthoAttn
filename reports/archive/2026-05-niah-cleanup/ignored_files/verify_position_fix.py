#!/usr/bin/env python3
"""Verify the double-increment position bug fix verification script."""

import torch
from src.dsra.mhdsra2.improved_dsra_mha import MHDSRA2Config, MultiHeadDSRA2

def test_position_double_increment():
    """Test that position increments correctly (not double increment)."""
    print("Testing MHDSRA2 position increment bug fix verification")
    print("=" * 60)
    
    torch.manual_seed(0)
    dim = 32
    heads = 4
    slots = 8
    
    cfg = MHDSRA2Config(
        dim=dim,
        heads=heads,
        slots=slots,
        read_topk=3,
        write_topk=2,
        local_window=6,
        use_local=True,
        use_retrieval=True,
        detach_state=False,
    )
    layer = MultiHeadDSRA2(cfg)
    
    # First chunk: 4 tokens
    x1 = torch.randn(2, 4, dim)
    # Second chunk: 5 tokens
    x2 = torch.randn(2, 5, dim)
    
    # Test state 1
    print(f"\nFirst forward pass with 4 tokens")
    _, state1 = layer(x1)
    print(f"  state1.position = {state1.position} (expected: 4)")
    
    # Test state 2
    print(f"\nSecond forward pass with 5 tokens")
    _, state2 = layer(x2, state=state1)
    print(f"  state2.position = {state2.position} (expected: 9)")
    
    # Verify
    assert state1.position == 4, f"Expected position1 {state1.position} should be 4"
    assert state2.position == 9, f"Expected position2 {state2.position} should be 9"
    
    print("\n✅ SUCCESS: Positions increment correctly (no double increment!)")
    
    return True

if __name__ == "__main__":
    test_position_double_increment()
