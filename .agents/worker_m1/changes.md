# Milestone 1 Code Modifications Summary

## Overview
Implemented Stanford Zoology MQAR (Multi-Query Associative Recall) domain specification alignment, dynamic vocabulary pool scaling, comprehensive boundary validation, device flexibility, `MQAROracleModel` (Ground Truth Oracle Probe), and expanded exhaustive unit test suite.

## Modified Files
1. `src/dsra/domain/mqar.py`
   - **Spec Alignment & Vocabulary Partitioning**:
     - Partitioned vocabulary into four strictly disjoint subspaces:
       - `[0]`: Pad / Loss Mask (`ignore_index=0`)
       - `[1 .. 1+key_pool_size)`: Keys Candidate Pool
       - `[val_start .. val_end)`: Values Candidate Pool
       - `[filler_start .. vocab_size)`: Fillers / Distractors Candidate Pool
     - Dynamically scaled `key_pool_size` and `val_pool_size` based on `vocab_size` (supporting $V=4, 32, 64, 8192$), resolving previous crashes when $V < 64$ while supporting explicit user overrides.
   - **Comprehensive Validation in `MQARConfig.__post_init__`**:
     - Enforced $V \ge 4$.
     - Enforced $K \ge 1, Q \ge 1, Q \le K$.
     - Enforced sequence length $L \ge 2K + Q$.
     - Enforced `key_pool_size + val_pool_size + 2 <= vocab_size`.
     - Enforced `num_kv_pairs <= key_pool_size` and `num_kv_pairs <= val_pool_size`.
     - Validated `insert_mode in ("uniform", "random")`.
   - **Data Generation Enhancements in `generate_mqar_batch`**:
     - Supported both `generate_mqar_batch(batch_size, config, ...)` and `generate_mqar_batch(config, batch_size=...)` calling conventions.
     - Robust `device` handling accepting `str` (e.g. `'cuda:0'`, `'cpu'`) and `torch.device`.
     - Seed isolation using `torch.Generator` without polluting global RNG state.
     - Implemented both `"uniform"` and `"random"` non-overlapping KV pair and query placement modes.
     - Enforced strictly causal autoregressive Next-Token target alignment ($Y[b, qpos_i] = \text{val}_i$) with zero future leakage, and $Y=0$ on all non-query positions (`ignore_index=0`).
   - **Ground Truth Oracle Model (`MQAROracleModel`)**:
     - Pure causal prefix KV memory table lookup inheriting from `torch.nn.Module`.
     - Scans sequence causally ($t \le \text{current\_time}$) to register $(k_i, v_i)$ prefix pairs.
     - Outputs high logit scale ($+100.0$) at true value token dimensions during query steps, achieving exact 100.0% accuracy and 0.0 cross-entropy loss.
   - **Docstrings & Comments**: Complete Chinese & English docstrings adhering to AGENTS.md rules.

2. `tests/test_mqar_data_generation.py`
   - Added exhaustive unit tests covering:
     * `test_mqar_config_validation_valid`: Basic config validation and dynamic defaults.
     * `test_mqar_config_validation_errors`: 8 boundary error test cases ($V<4$, $K<1$, $Q<1$, $Q>K$, short $L$, invalid insert_mode, pool overflows).
     * `test_mqar_dynamic_vocab_scaling`: Small ($V=32$), medium ($V=64$), large ($V=8192$), and minimal ($V=4$) vocabulary testing.
     * `test_generate_mqar_batch_shapes_and_values`: Shape, device, value bounds, and target consistency.
     * `test_vocabulary_partitioning_disjointness`: Verification that Pad, Keys, Values, and Fillers are pairwise disjoint with zero overlap.
     * `test_causal_key_value_placement_and_zero_future_leakage`: Causal ordering and prevention of answer leakage in next-token prompt.
     * `test_insert_mode_uniform_and_random`: Both uniform and randomized position allocations.
     * `test_device_flexibility_and_string_argument`: Verification of string vs torch.device handling.
     * `test_generator_seed_reproducibility`: Deterministic seed reproducibility and isolation.
     * `test_mqar_oracle_model_100_percent_accuracy_and_zero_loss`: Exact 100.0% accuracy and 0.0 loss verification across multiple scales ($L \in [64, 1024]$, $K \in [2, 16]$, $V \in [32, 1024]$, $Q \le K$).

## Verification Summary
- `pytest tests/test_mqar_data_generation.py -v`: 10/10 passed (100%).
- `pytest tests/ -v`: 418/418 passed across all 13 test files (100%).
- `ruff check src/dsra/domain/mqar.py tests/test_mqar_data_generation.py`: 0 violations.
