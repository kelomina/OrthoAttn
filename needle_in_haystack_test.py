"""Compatibility wrapper for legacy imports and CLI entrypoints."""

from scripts.needle_in_haystack_test import *  # noqa: F401,F403

if __name__ == "__main__":
    from runpy import run_module

    run_module("scripts.needle_in_haystack_test", run_name="__main__")
