# DSRA Attention

DSRA Attention is a research-oriented verification suite for streaming long-sequence
attention. It explores whether a model can keep useful long-range information with
a fixed-size differentiable state store, semantic routing, orthogonal updates, and
optional exact memory retrieval.

In plain terms, this project is trying to answer a practical question: when the input
sequence becomes very long, can an attention layer keep the important facts without
paying the full cost of looking back at every previous token? DSRA/MHDSRA2 treats the
model's memory a bit like a compact notebook: recent details stay in a local window,
older high-level information is compressed into reusable slots, and exact memories
can be paged back when a task needs precise recall.

## Project Status

This repository is an experimental research codebase, not a production LLM library.
The implementation, tests, and reports are organized so that new attention variants
can be compared, diagnosed, and reproduced.

Current focus areas:

- **MHDSRA2 core implementation**: multi-head streaming attention with slot, local,
  and retrieval branches.
- **Paged exact memory**: CPU-side key/value memory used to retrieve distant tokens
  without keeping the whole sequence on GPU.
- **Arithmetic emergence diagnostics**: small controlled tasks for checking whether
  layer depth and training settings help arithmetic rules emerge.
- **Needle-in-a-haystack diagnostics**: long-context recall tests used to expose
  memory routing and gradient-flow limits.
- **Benchmark and report generation**: reproducible scripts that write Markdown and
  JSON outputs under `reports/`.

Known limitations:

- Long-context results are still diagnostic. Some NIAH experiments show partial
  progress, but this project should not be read as claiming solved long-context
  reasoning.
- Several experiment scripts are intentionally research-heavy and may require CUDA
  hardware for practical runtimes.
- No open-source license file is currently present. Add a `LICENSE` file before
  publishing or accepting external contributions.

## Features

- Fixed-capacity differentiable state slots for streaming attention.
- Orthogonal incremental updates to reduce state saturation.
- Three-way MHDSRA2 fusion:
  - `slot`: compressed global memory.
  - `local`: sliding-window recent context.
  - `retrieval`: optional paged exact memory.
- Position encoding modes: `none`, `rope`, `alibi`, and `timestamps`.
- Compatibility wrappers for legacy imports and `python main.py ...` commands.
- Unit tests, smoke tests, benchmark scripts, diagnostic grids, and report writers.

## Repository Layout

```text
.
|-- config/                  # Experiment configuration objects
|-- docs/                    # Local case studies and experiment notes
|-- reports/                 # Generated Markdown/JSON reports
|-- scripts/                 # CLI entrypoints, benchmarks, diagnostics
|-- src/dsra/                # Formal Python package
|   |-- application/         # Use-case services and unit-of-work boundary
|   |-- domain/              # Domain specs and validation objects
|   |-- infrastructure/      # Repository implementations, paged memory adapters
|   `-- mhdsra2/             # MHDSRA2 attention and exact memory engine
|-- tests/                   # Unit, integration, smoke, and report tests
|-- main.py                  # Legacy-compatible CLI wrapper
`-- pyproject.toml           # Package metadata and development dependencies
```

Root-level files such as `dsra_layer.py`, `dsra_model.py`, and
`needle_in_haystack_test.py` are compatibility wrappers. New code should usually
import from `src/dsra/` or call scripts under `scripts/`.

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
specific. If GPU support is required, install the PyTorch build that matches your
CUDA driver from the official PyTorch instructions before running heavy experiments.

## Quick Start

Show the unified CLI help:

```bash
python main.py -h
```

Run the full unit test suite:

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

Generate only the report index without running experiments:

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

## Common Commands

```bash
# Unit tests discovered from tests/
python main.py unit

# Complexity and performance benchmark
python main.py benchmark

# Associative recall toy task
python main.py recall

# Needle-in-a-haystack diagnostic
python main.py needle

# JSON retrieval diagnostic
python main.py json_retrieval

# MHDSRA2 verification
python main.py mhdsra2

# MHDSRA2 vs DSRA comparison
python main.py mhdsra2_compare

# Arithmetic layer-emergence report
python main.py mhdsra2_layer_emergence

# Carry-rule diagnostic grid, resumable and potentially long-running
python main.py mhdsra2_carry_diagnostic_grid

# Ablation study
python main.py ablation
```

Some commands write outputs to `reports/`. Long-running commands may also create
checkpoint files so interrupted experiments can resume.

## Reports

The `reports/` directory is the canonical place for generated experiment artifacts.
Tracked reports should usually include both:

- a human-readable `.md` summary;
- a machine-readable `.json` payload when the script produces structured data.

The unified `all` command writes:

- `reports/all_output.txt`: captured terminal output;
- `reports/run_summary.md`: summary of executed suites and generated files.

Use `reports/archive/` or other ignored local folders for temporary diagnostic
snapshots that are not meant to become canonical results.

## Architecture Overview

The codebase follows a small domain-driven layout:

- **Domain layer** (`src/dsra/domain/`): validates model and attention specs.
- **Application layer** (`src/dsra/application/`): coordinates forward-call state,
  retrieval services, and experiment use cases.
- **Infrastructure layer** (`src/dsra/infrastructure/`): provides concrete memory
  and report repositories.
- **Core implementation layer** (`src/dsra/mhdsra2/`): implements `MultiHeadDSRA2`
  and `PagedExactMemory`.
- **Compatibility layer** (`src/dsra/dsra_layer.py`, `src/dsra/dsra_model.py`, and
  root wrappers): keeps older imports and CLI commands working.

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

Tests should be deterministic and should not depend on production services, real
API keys, or unstable network calls.

## Development Notes

- Prefer modifying existing modules over creating parallel implementations.
- Keep public function names, argument order, return types, and exception behavior
  stable unless a migration is documented.
- Use `cuda:0` explicitly when CUDA is available and a script needs GPU execution.
- Keep tests under `tests/`, reports under `reports/`, and configuration under
  `config/`.
- Do not commit real secrets, tokens, passwords, or private datasets.
- If an experiment fails, record the failure and avoid turning temporary tuning
  into a hidden permanent behavior change.

## Current Research Notes

Recent MHDSRA2 arithmetic diagnostics suggest that lower learning rates and enough
per-stage training budget can stabilize simple carry-rule learning, but two-digit
rule composition and out-of-distribution arithmetic such as `100+100=200` still need
more evidence before being treated as solved.

Recent NIAH case studies show that disabling state detachment can improve gradient
flow, but high-cardinality long-context recall remains a hard diagnostic target. Use
these experiments as evidence for research direction, not as production guarantees.

## Contributing

This repository does not yet include a formal `CONTRIBUTING.md`. Until one is added,
please keep contributions small and evidence-based:

1. Open with the problem being solved and the expected measurable improvement.
2. Add or update tests for the changed behavior.
3. Run the relevant commands locally and report exact results.
4. Keep generated report changes only when the result or schema intentionally changes.

## License

No license file is currently included. If this repository is published as an open
source project, add a clear license such as MIT, Apache-2.0, or another license that
matches the intended reuse policy.
