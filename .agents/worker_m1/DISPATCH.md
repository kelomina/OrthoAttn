## 2026-08-22T09:03:37Z

Milestone 1: MQAR Domain Spec Alignment, Edge Cases & Ground Truth Oracle Probe.
Working directory: E:/Project/python/DSRA/.agents/worker_m1
Project root: E:/Project/python/DSRA
Original Request: E:/Project/python/DSRA/ORIGINAL_REQUEST.md
User Rules: E:/Project/python/DSRA/AGENTS.md
Project Plan: E:/Project/python/DSRA/PROJECT.md

Tasks:
1. In `src/dsra/domain/mqar.py`:
   - Refine `MQARConfig` and `generate_mqar_batch` to strictly align with Stanford Zoology (ICLR 2024 / HazyResearch `zoology.data.associative_recall`) specification.
   - Dynamic vocabulary pool scaling: dynamically compute `key_pool_size`, `val_pool_size`, and `filler_start` based on `vocab_size` without crashing when $V < 64$ (support $V=32, 64, 8192$, etc.). Allow user override via `MQARConfig(key_pool_size=..., val_pool_size=...)`.
   - Comprehensive validation in `MQARConfig.__post_init__`: ensure $K \ge 1$, $Q \ge 1$, $Q \le K$, $V \ge 4$, $L \ge 2K + Q$, `key_pool_size + val_pool_size + 2 <= vocab_size`.
   - Robust `device` handling: accept both `torch.device` and `str` (e.g. `'cuda:0'`, `'cpu'`), correctly resolving `torch.device(config.device)`.
   - Implement `MQAROracleModel(torch.nn.Module)`: a clean, pure causal prefix KV lookup model that reads input sequences, extracts causal (key, value) pairs seen in the prefix prior to each query position, and outputs logits putting extreme probability on the ground truth value for query positions, achieving exact 100.0% accuracy ($loss = 0.0$).
   - Ensure full Chinese & English docstrings adhering to AGENTS.md rules.
2. In `tests/test_mqar_data_generation.py`:
   - Add/update comprehensive unit tests covering:
     * Standard MQAR data generation and tensor shapes.
     * Vocabulary partitioning disjointness (Pad=0, Keys, Values, Fillers).
     * Causal key-value placement and zero future leakage.
     * Target loss masking (`ignore_index=0` on all non-query positions).
     * Boundary conditions ($V=32, 64, 8192$, $Q < K$, $Q = K$).
     * Parameter validation error tests (e.g., $Q > K$, invalid $V$, invalid sequence length).
     * Device placement test (`cuda:0` if available, `cpu`).
     * Oracle model verification: `MQAROracleModel` achieves exact 100.0% accuracy and loss = 0.0 on test batches.
3. Verification:
   - Run `pytest tests/test_mqar_data_generation.py -v` and `pytest tests/` to ensure all tests pass 100%.
   - Ensure no regressions across the repository.
