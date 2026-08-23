## 2026-08-22T00:55:30Z
You are the Stanford Zoology MQAR Specification Miner.
Working directory: E:/Project/python/DSRA/.agents/spec_miner_survey
Original Request: E:/Project/python/DSRA/ORIGINAL_REQUEST.md
User Rules: E:/Project/python/DSRA/AGENTS.md

Your task is to conduct a thorough read-only specification investigation on:
1. `src/dsra/domain/mqar.py` and any related domain/data generation files in the repository.
2. Compare line-by-line against Stanford Zoology (ICLR 2024 / HazyResearch `zoology.data.associative_recall`) MQAR specification:
   - Vocabulary partitioning: Pad/Ignore = [0], Keys = [1..K_pool], Values = [K_pool+1..V_pool], Fillers = [V_pool+1..vocab_size-1], strictly disjoint.
   - Key-value distribution: K pairs of (k_i, v_i) randomly inserted in the prefix, separated by random distractor/filler tokens.
   - Query generation: Q query keys randomly permuted in the suffix, strictly autoregressively causal (input X[qpos]=Key, target Y[qpos]=Value), no future information泄漏/leakage.
   - Loss masking: Target Y = 0 (ignore_index = 0) at all positions except query prediction positions.
   - Mathematical equivalence & parameters (vocab_size, num_kv_pairs, num_queries, seq_len, etc.).
3. Identify all discrepancies, potential bugs, missing features, edge cases, and required fixes.

Output requirements:
- Write your detailed findings and evidence to `E:/Project/python/DSRA/.agents/spec_miner_survey/report.md`
- Write `E:/Project/python/DSRA/.agents/spec_miner_survey/handoff.md`
- When done, send a message to parent orchestrator with the summary and path.
- DO NOT edit any source code. You are read-only.
