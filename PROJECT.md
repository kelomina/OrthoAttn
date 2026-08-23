# Project: DSRA Stanford Zoology MQAR Alignment & Verification

## Architecture
- **Domain Layer (`src/dsra/domain/mqar.py`)**: Defines `MQARConfig`, `generate_mqar_batch`, `evaluate_mqar_predictions`, and `MQAROracleModel` (Ground Truth Oracle probe). Fully aligned with Stanford Zoology (ICLR 2024 / HazyResearch `zoology.data.associative_recall`) specification.
- **Benchmark / Script Layer (`scripts/benchmark_mqar.py`)**: End-to-end benchmark execution script comparing Ground Truth Oracle, Standard Causal Transformer (`StandardAttentionLM`), and `MultiLayerMHDSRA2Model` on MQAR tasks ($L=512, K=4$; $L=1024, K=8$; $L=2048, K=16$).
- **Test Layer (`tests/test_mqar_data_generation.py`)**: Exhaustive unit and boundary test suite covering vocabulary partitioning, causal key-value alignment, loss masking, edge cases ($V < 64, V \ge 512, Q=K, Q < K$), and Oracle 100% accuracy probe.
- **Reports Layer (`reports/`)**: Formal validation results in `reports/mqar_benchmark_results.json` and `reports/mqar_benchmark_validation_report.md`.

## Code Layout
- `src/dsra/domain/mqar.py`: Domain data generation, causal alignment, loss masking, Oracle lookup model.
- `scripts/benchmark_mqar.py`: Benchmark runner, training loops, Transformer & DSRA comparative evaluation.
- `tests/test_mqar_data_generation.py`: Test suite for data generator, boundaries, and Oracle verification.
- `reports/mqar_benchmark_results.json`: Benchmark raw numerical records.
- `reports/mqar_benchmark_validation_report.md`: Formal verification and audit report.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Spec Alignment & Vocabulary Partitioning | Pad=[0], Keys=[1..K_pool], Values=[K_pool+1..V_pool], Fillers=[V_pool+1..V-1], strictly disjoint, dynamic scaling | M1 | ORIGINAL_REQUEST R1 |
| 2 | Causal KV Placement & Loss Masking | Prefix KV pairs with distractor fillers, suffix shuffled queries, strictly causal autoregressive target Y with ignore_index=0 | M1 | ORIGINAL_REQUEST R1 |
| 3 | Robust Parameter Validation & Device Handling | Support arbitrary V, validation for Q <= K and K >= 1, support both string ('cuda:0', 'cpu') and torch.device | M1 | Spec Survey Report |
| 4 | Ground Truth Oracle Lookup Probe | Pure causal lookup model verifying pipeline logic, achieving exact 100.0% accuracy and 0.0 loss | M1 | ORIGINAL_REQUEST R3 |
| 5 | Standard Causal Transformer Baseline | Pre-LN + RoPE + SDPA 2-layer Transformer achieving 90%+ (100%) accuracy on L=512, K=4 and L=1024, K=8 | M2 | ORIGINAL_REQUEST R4 |
| 6 | Unified Benchmark & Evaluation Pipeline | Zero dummy code, authentic training/eval loop comparing Oracle, Transformer, and MHDSRA2 on cuda:0 | M2 | ORIGINAL_REQUEST R2 & R4 |
| 7 | Full Test Suite 100% Pass | All repository unit tests including test_mqar_data_generation pass 100% | M3 | Acceptance Criteria |
| 8 | Formal Markdown/JSON Validation Report | Complete formal audit, Oracle verification, and baseline comparison reports in reports/ | M3 | Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Domain Spec Alignment, Edge Cases & Oracle Probe | `src/dsra/domain/mqar.py`, `tests/test_mqar_data_generation.py`: Spec alignment, dynamic vocab scaling, parameter validation, device handling, and `MQAROracleModel` with 100% accuracy probe. | None | DONE |
| M2 | Standard Transformer Baseline & Benchmark Runner | `scripts/benchmark_mqar.py`, `reports/mqar_benchmark_results.json`: `StandardAttentionLM` baseline, unified benchmark CLI, LR scheduling, execution on L=512 & L=1024 on cuda:0. | M1 | DONE |
| M3 | Full Suite Verification, Audit & Formal Report | Comprehensive testing (all repo tests 100% pass), Reviewer/Challenger/Auditor gates, and `reports/mqar_benchmark_validation_report.md` generation. | M1, M2 | DONE |

## Interface Contracts
### `MQARConfig` & `generate_mqar_batch`
```python
@dataclass
class MQARConfig:
    vocab_size: int = 8192
    num_kv_pairs: int = 8
    num_queries: int = 8
    seq_len: int = 1024
    key_pool_size: Optional[int] = None
    val_pool_size: Optional[int] = None
    device: Union[str, torch.device] = "cpu"
    seed: Optional[int] = None

def generate_mqar_batch(config: MQARConfig, batch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns (X, Y) with shape (B, L), dtype torch.long
```

### `MQAROracleModel`
```python
class MQAROracleModel(torch.nn.Module):
    def __init__(self, vocab_size: int, pad_token_id: int = 0): ...
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Returns logits with shape (B, L, vocab_size) where query positions place infinite/high logit on true value
```

### `StandardAttentionLM` / Baseline Model
```python
class StandardAttentionLM(torch.nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, num_layers: int = 2, num_heads: int = 4, max_seq_len: int = 2048): ...
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Returns logits with shape (B, L, vocab_size)
```
