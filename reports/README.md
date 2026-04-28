# Reports Directory Policy

This directory stores benchmark and diagnostic outputs that are useful for review.

## Tracked Official Reports

Files directly under `reports/` are treated as official artifacts when they are
stable and intentionally generated for comparison or publication. Current tracked
examples include:

- `mhdsra2_vs_dsra_compare.{json,md}`: lightweight MHDSRA2 vs DSRA comparison.
- `mhdsra2_vs_dsra_next_round_benchmark.{json,md}`: unified next-round benchmark.
- `mhdsra2_vs_dsra_paper_style_summary.md`: paper-style summary.
- `retrieval_tau_ablation.{json,md}`: retrieval tau ablation report.
- `run_summary.md` and `all_output.txt`: top-level run summaries.

When adding a new official report, commit the `.md` summary and the matching
machine-readable `.json` payload when available.

## Local Archives

The following directories are local-only and ignored by Git:

- `reports/archive/`
- `reports/paper_seed_*/`

Use `reports/archive/` for temporary diagnostic snapshots that may help local
inspection but should not become canonical project artifacts. Use
`reports/paper_seed_*/` for imported or regenerated seed-specific comparison
bundles used as local baselines.

## Temporary Logs

Do not commit transient run logs, empty error files, or launch-test files such as:

- `*.log`
- `*.err`
- `_start_process_quote_test.*`

If a log contains evidence that should be preserved, summarize the important
result in a tracked `.md` report instead of committing the raw log.

## Regeneration Notes

Tests may rewrite some tracked reports while validating report generation. If a
test run changes a report unintentionally, inspect the diff first. Keep the diff
only when the benchmark result or report schema is intentionally updated;
otherwise restore it before committing.
