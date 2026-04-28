# Retrieval Tau Ablation

## Scope
- Model: `MH-DSRA-v2 (paged recall)` only.
- Cases: full Diagnostic A/B/C case grids used by the CUDA reports.
- Purpose: check whether `retrieval_tau=8.0` is a narrow single-point fit or part of a stable plateau.

## Summary Table

| retrieval_tau | A exact_match_rate | B latest_fact_accuracy | B stale_fact_rate | C counterexample_accuracy | C majority_trap_rate |
|---:|---:|---:|---:|---:|---:|
| 4.0 | 1.0000 | 1.0000 | 0.0000 | 0.3333 | 0.6667 |
| 6.0 | 1.0000 | 1.0000 | 0.0000 | 0.6667 | 0.3333 |
| 8.0 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| 10.0 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |

## Per-case Results

### retrieval_tau=4.0

#### A exact recall

| Case | Predicted | Target | Correct | Confidence margin |
|---|---:|---:|---:|---:|
| exact_recall.early | 16 | 16 | 1.0 | 1.287820 |
| exact_recall.middle | 16 | 16 | 1.0 | 1.287820 |
| exact_recall.late | 16 | 16 | 1.0 | 1.308346 |

#### B latest fact

| Case | Predicted | Target | Correct | Confidence margin |
|---|---:|---:|---:|---:|
| error_override.gap_128 | 1 | 1 | 1.0 | 0.089921 |
| error_override.gap_1024 | 1 | 1 | 1.0 | 0.086940 |
| error_override.gap_4096 | 1 | 1 | 1.0 | 0.078300 |

#### C anti-fixation

| Case | Predicted | Target | Correct | Confidence margin |
|---|---:|---:|---:|---:|
| anti_fixation.distractors_16 | 1 | 1 | 1.0 | 1.124193 |
| anti_fixation.distractors_64 | 0 | 1 | 0.0 | 0.047300 |
| anti_fixation.distractors_256 | 0 | 1 | 0.0 | 0.673779 |

### retrieval_tau=6.0

#### A exact recall

| Case | Predicted | Target | Correct | Confidence margin |
|---|---:|---:|---:|---:|
| exact_recall.early | 16 | 16 | 1.0 | 1.287820 |
| exact_recall.middle | 16 | 16 | 1.0 | 1.287820 |
| exact_recall.late | 16 | 16 | 1.0 | 1.308346 |

#### B latest fact

| Case | Predicted | Target | Correct | Confidence margin |
|---|---:|---:|---:|---:|
| error_override.gap_128 | 1 | 1 | 1.0 | 0.089921 |
| error_override.gap_1024 | 1 | 1 | 1.0 | 0.086940 |
| error_override.gap_4096 | 1 | 1 | 1.0 | 0.078300 |

#### C anti-fixation

| Case | Predicted | Target | Correct | Confidence margin |
|---|---:|---:|---:|---:|
| anti_fixation.distractors_16 | 1 | 1 | 1.0 | 1.403347 |
| anti_fixation.distractors_64 | 1 | 1 | 1.0 | 0.429503 |
| anti_fixation.distractors_256 | 0 | 1 | 0.0 | 0.267444 |

### retrieval_tau=8.0

#### A exact recall

| Case | Predicted | Target | Correct | Confidence margin |
|---|---:|---:|---:|---:|
| exact_recall.early | 16 | 16 | 1.0 | 1.287820 |
| exact_recall.middle | 16 | 16 | 1.0 | 1.287820 |
| exact_recall.late | 16 | 16 | 1.0 | 1.308346 |

#### B latest fact

| Case | Predicted | Target | Correct | Confidence margin |
|---|---:|---:|---:|---:|
| error_override.gap_128 | 1 | 1 | 1.0 | 0.089921 |
| error_override.gap_1024 | 1 | 1 | 1.0 | 0.086940 |
| error_override.gap_4096 | 1 | 1 | 1.0 | 0.078300 |

#### C anti-fixation

| Case | Predicted | Target | Correct | Confidence margin |
|---|---:|---:|---:|---:|
| anti_fixation.distractors_16 | 1 | 1 | 1.0 | 1.554294 |
| anti_fixation.distractors_64 | 1 | 1 | 1.0 | 0.790917 |
| anti_fixation.distractors_256 | 1 | 1 | 1.0 | 0.097975 |

### retrieval_tau=10.0

#### A exact recall

| Case | Predicted | Target | Correct | Confidence margin |
|---|---:|---:|---:|---:|
| exact_recall.early | 16 | 16 | 1.0 | 1.287820 |
| exact_recall.middle | 16 | 16 | 1.0 | 1.287820 |
| exact_recall.late | 16 | 16 | 1.0 | 1.308346 |

#### B latest fact

| Case | Predicted | Target | Correct | Confidence margin |
|---|---:|---:|---:|---:|
| error_override.gap_128 | 1 | 1 | 1.0 | 0.089921 |
| error_override.gap_1024 | 1 | 1 | 1.0 | 0.086940 |
| error_override.gap_4096 | 1 | 1 | 1.0 | 0.078300 |

#### C anti-fixation

| Case | Predicted | Target | Correct | Confidence margin |
|---|---:|---:|---:|---:|
| anti_fixation.distractors_16 | 1 | 1 | 1.0 | 1.629961 |
| anti_fixation.distractors_64 | 1 | 1 | 1.0 | 1.031026 |
| anti_fixation.distractors_256 | 1 | 1 | 1.0 | 0.391253 |

## Conclusion

`retrieval_tau=8.0` sits inside a stable passing range: [8.0, 10.0]. The result is therefore not a single-value fit to the current A/B/C probes.
