"""Compatibility wrapper for legacy imports and CLI entrypoints."""

from scripts.attention_family_benchmark import *  # noqa: F401,F403

if __name__ == "__main__":
    from runpy import run_module

    run_module("scripts.attention_family_benchmark", run_name="__main__")
