"""Compatibility wrapper for legacy imports and CLI entrypoints."""

from scripts.benchmark_complexity import *  # noqa: F401,F403

if __name__ == "__main__":
    from runpy import run_module

    run_module("scripts.benchmark_complexity", run_name="__main__")
