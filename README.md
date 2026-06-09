# DSRA Attention

DSRA Attention is a research-oriented verification suite for streaming
long-sequence attention. The active architecture in this repository is
MHDSRA2: a multi-head attention layer that combines compact slot memory, a
bounded local window, and optional CPU-side exact retrieval.

In plain terms, this project is trying to answer a practical question: when an
input sequence becomes very long, can an attention layer keep useful facts
without paying the full cost of looking back at every previous token? MHDSRA2
treats memory like a compact notebook. Recent details stay in a local window,
older information is blended into reusable slots, and exact token memories can
be paged back from CPU memory when a task needs precise recall.

## Project Status

This repository is an experimental research codebase, not a production LLM
library. It is organized for diagnosis, ablation studies, and reproducible
comparison of attention variants.

Current focus areas:

- **MHDSRA2 core implementation**: slot, local, and retrieval branches fused by
  a learned gate.
- **Batch-isolated paged exact memory**: CPU-side key/value pages with per-sample
  isolation, retrieval masks, future-token cutoffs, optional page caps, and
  profiling hooks.
- **Retrieval quality diagnostics**: smoke tests and ablations for exact
  retrieval quality, batch/loop equivalence, latest-wins recall, and
  cross-sample leak prevention.
- **Evidence-supervised retrieval experiments**: opt-in NIAH/JSON auxiliary
  losses and an opt-in zero-initialized retrieval gate adapter. These are
  disabled by default and are still experimental.
- **Arithmetic and JSON retrieval diagnostics**: controlled tasks used to test
  generalization, validation/test separation, and training stability.
- **Benchmark and report generation**: scripts that write Markdown and JSON
  outputs under `reports/`.
- **Security and reproducibility guardrails**: report path validation, disabled
  SwanLab uploads by default, pinned WikiText dataset revisions for fresh
  downloads, and an OSV dependency audit script.

Known limitations:

- Long-context results are diagnostic. This repository should not be read as
  claiming solved long-context reasoning.
- Some NIAH runs still show weak or zero validation accuracy at higher-cardinality
  settings, even when synthetic exact retrieval smoke tests pass.
- The current slot update is a gated blended update with overwrite diagnostics;
  it is not a strict orthogonal projection update.
- CPU-side paged memory is a reference implementation. It is useful for exact
  retrieval experiments, but it is not a FAISS/ScaNN-grade production index.
- Several experiment scripts are research-heavy and may require `cuda:0` for
  practical runtimes.
- `docs/` is ignored in this checkout. Local case-study notes may exist there,
  but they are not guaranteed to be part of a clean clone.

## Features

- Fixed-capacity differentiable state slots for streaming attention.
- Slot/local/retrieval MHDSRA2 fusion:
  - `slot`: compressed global memory.
  - `local`: bounded sliding-window recent context.
  - `retrieval`: optional CPU-side paged exact memory.
- Overwrite-aware gated slot updates with diagnostics such as novelty,
  overwrite gate, write drive, usage, and confidence.
- Batch-isolated external memory for batch sizes greater than one.
- Retrieval metadata, validity masks, per-sample `max_position`, and selected
  auxiliary outputs for evidence supervision.
- Legacy-compatible APIs for older DSRA imports and `python main.py ...`
  commands.
- Unit tests, smoke tests, benchmark scripts, diagnostic grids, dependency
  auditing, and report writers.

## Repository Layout

```text
.
|-- archive/                 # Historical snapshots and old local copies
|-- config/                  # Experiment configuration objects
|-- reports/                 # Generated Markdown/JSON reports and local logs
|-- scripts/                 # CLI entrypoints, benchmarks, diagnostics, audits
|-- src/dsra/                # Formal Python package
|   |-- application/         # Use-case services and unit-of-work boundary
|   |-- domain/              # Specs, validation, and model-name normalization
|   |-- infrastructure/      # Repository implementations and report adapters
|   `-- mhdsra2/             # MHDSRA2 attention and exact memory engine
|-- tests/                   # Unit, integration, smoke, and report tests
|-- main.py                  # Legacy-compatible CLI wrapper
`-- pyproject.toml           # Package metadata and development dependencies
```

The root `main.py` file is a compatibility wrapper around `scripts/main.py`.
The older DSRA public surface is now implemented inside `src/dsra/dsra_layer.py`
and `src/dsra/dsra_model.py`. New code should usually import from `src/dsra/`
or call scripts under `scripts/`.

The model name `dsra` is currently treated as an archived alias for `mhdsra2`.
Do not interpret `dsra` vs `mhdsra2` report labels as two independent active
architectures unless a future change reintroduces a separate DSRA builder.

## Installation

Python 3.10 or newer is required.

On Windows PowerShell:

```powershell
python -m venv .env
.\.env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On bash/zsh:

```bash
python -m venv .env
source .env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

PyTorch is listed as a dependency, but CUDA builds of PyTorch can be environment
specific. If GPU support is required, install the PyTorch build that matches
your CUDA driver from the official PyTorch instructions before running heavy
experiments.

## Quick Start

Show the unified CLI help:

```bash
python main.py -h
```

Run tests discovered from `tests/`:

```bash
pytest
```

Run the project-compatible unit entrypoint:

```bash
python main.py unit
```

Run the MHDSRA2 smoke verification:

```bash
python main.py mhdsra2
```

Generate only the top-level report index:

```bash
python main.py report
```

Run a fast version of the `all` suite:

```powershell
$env:DSRA_FAST_ALL="1"; python main.py all
```

```bash
DSRA_FAST_ALL=1 python main.py all
```

## Unified CLI Commands

These commands are registered by `scripts/main.py` and can be called through the
root wrapper as `python main.py <command>`:

```bash
# Unit tests discovered from tests/
python main.py unit

# Complexity and performance benchmark
python main.py benchmark

# State saturation diagnostic
python main.py saturation

# Associative recall toy task
python main.py recall

# Needle-in-a-haystack diagnostic
python main.py needle

# Needle-in-a-haystack capacity reports
python main.py needle_capacity

# JSON retrieval diagnostic
python main.py json_retrieval

# JSON retrieval generalization diagnostic
python main.py json_retrieval_generalization

# Attention family benchmark
python main.py attention_family_benchmark

# MHDSRA2 verification
python main.py mhdsra2

# MHDSRA2 comparison report
python main.py mhdsra2_compare

# Unified next-round benchmark
python main.py next_round_benchmark

# Arithmetic layer-emergence report
python main.py mhdsra2_layer_emergence

# Curriculum strategy grid
python main.py mhdsra2_curriculum_strategy_grid

# Carry-rule diagnostic grid, resumable and potentially long-running
python main.py mhdsra2_carry_diagnostic_grid

# Ablation study
python main.py ablation

# Interactive chat entrypoint, if the required model assets are available
python main.py chat

# Generate reports/run_summary.md without running experiment suites
python main.py report
```

For smaller comparison workloads:

```powershell
$env:DSRA_FAST_COMPARE="1"; python main.py mhdsra2_compare
```

```bash
DSRA_FAST_COMPARE=1 python main.py mhdsra2_compare
```

Some commands write outputs to `reports/`. Long-running commands may also create
checkpoint files so interrupted experiments can resume.

## Standalone Scripts

Some current workflows are exposed as standalone scripts instead of unified
`main.py` commands:

```bash
# Installed-package OSV audit; sends package names and versions to OSV
python scripts/audit_installed_packages_osv.py --output reports/dependency_osv_audit.json --fail-on-vuln

# Exact retrieval quality smoke: batch isolation, future cutoff, latest-wins recall
python scripts/mhdsra2_batch_retrieval_quality_smoke.py

# Batched retrieval profiling
python scripts/mhdsra2_batched_retrieval_benchmark.py --json-out reports/mhdsra2_batched_retrieval_profile.json --markdown-out reports/mhdsra2_batched_retrieval_profile.md

# P0/P1 regression ablation for slot overwrite, page recall, and forward_step reuse
python scripts/mhdsra2_bugfix_ablation.py

# P2 engineering regression ablation
python scripts/mhdsra2_p2_engineering_ablation.py

# Unified MHDSRA2 quality-improvement ablation; dry-run shows planned rows only
python scripts/mhdsra2_quality_improvement_ablation.py --dry-run --device cpu
```

The quality-improvement ablation defaults to these current experimental groups:
`baseline`, `evidence_hit_supervision`, `learned_retrieval_gate`, and
`evidence_plus_gate`. Older exploratory groups such as `retrieval_query_pooling`,
`retrieval_gate_quality`, and `combined` still exist but should not be treated as
the default research direction.

## Reports

The `reports/` directory stores generated experiment artifacts. Files directly
under `reports/` may be treated as official artifacts only when they are stable
and intentionally generated for review or comparison.

Tracked official reports should usually include both:

- a human-readable `.md` summary;
- a machine-readable `.json` payload when the script produces structured data.

The unified `all` command writes:

- `reports/all_output.txt`: captured terminal output;
- `reports/run_summary.md`: summary of executed suites and generated files.

Temporary diagnostic snapshots, raw logs, and seed-specific bundles should stay
in ignored locations such as `reports/archive/` or other local-only paths unless
the result or schema intentionally becomes part of the canonical evidence.

## Architecture Overview

The codebase follows a small domain-driven layout:

- **Domain layer** (`src/dsra/domain/`): validates attention specs and normalizes
  archived model aliases.
- **Application layer** (`src/dsra/application/`): coordinates forward-call state,
  retrieval services, model factories, and arithmetic experiment services.
- **Infrastructure layer** (`src/dsra/infrastructure/`): provides paged memory
  and JSON retrieval report repositories.
- **Core implementation layer** (`src/dsra/mhdsra2/`): implements
  `MultiHeadDSRA2`, `MHDSRA2Config`, `MHDSRA2State`, and `PagedExactMemory`.
- **Compatibility layer** (`src/dsra/dsra_layer.py`, `src/dsra/dsra_model.py`,
  and root `main.py`): keeps older imports and CLI commands working.

## Testing and Quality

Recommended local checks before committing:

```bash
pytest
ruff check .
```

For a faster smoke check:

```bash
python main.py unit
python main.py mhdsra2
```

Focused checks used by recent MHDSRA2 retrieval work include:

```bash
pytest tests/test_memory_lifecycle_regressions.py tests/test_multilayer_retrieval_regressions.py -q
pytest tests/test_diagnostic_gate_policy_regressions.py -q
pytest tests/test_mhdsra2_quality_improvement_ablation.py -q
pytest tests/test_security_regressions.py -q
```

Tests should be deterministic and should not depend on production services, real
API keys, or unstable network calls.

## Development Notes

- Prefer modifying existing modules over creating parallel implementations.
- Keep public function names, argument order, return types, and exception
  behavior stable unless a migration is documented.
- Use `cuda:0` explicitly when CUDA is available and a script needs GPU
  execution. Some config helpers may still accept `auto`, but CUDA experiments
  should resolve to `cuda:0`.
- Keep tests under `tests/`, reports under `reports/`, and configuration under
  `config/`.
- Do not commit real secrets, tokens, passwords, or private datasets.
- SwanLab logging is disabled by default in current scripts. Use cloud logging
  only when explicitly intended.
- If an experiment fails, record the failure and avoid turning temporary tuning
  into a hidden permanent behavior change.

## Current Research Notes

Recent exact-retrieval smoke tests show that batch-isolated paged memory can
retrieve the intended synthetic token, preserve latest-wins behavior, avoid
cross-sample leakage, and respect future-token cutoffs. This validates the
external memory plumbing, not end-to-end task competence.

Recent quality-improvement ablations show that NIAH and JSON generation remain
diagnostic targets. In the available report artifacts, exact retrieval smoke
passes, but NIAH validation accuracy remains weak in several rows and JSON exact
match remains zero for the sampled small datasets. Treat retrieval changes as
research hypotheses unless validation improvements are clear and reproducible.

Arithmetic diagnostic reports can show strong in-distribution two-digit results
under controlled settings, while headline or out-of-distribution arithmetic may
still fail. Do not treat those diagnostics as evidence of general arithmetic
reasoning.

## Contributing

This repository does not yet include a formal `CONTRIBUTING.md`. Until one is
added, keep contributions small and evidence-based:

1. Open with the problem being solved and the expected measurable improvement.
2. Add or update tests for the changed behavior.
3. Run the relevant commands locally and report exact results.
4. Keep generated report changes only when the result or schema intentionally
   changes.

## License

This project is licensed under the Apache License 2.0. When distributing or
publishing the repository, keep the Apache-2.0 license notice and related
metadata in sync with the source tree.
