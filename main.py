"""Compatibility wrapper for legacy imports and CLI entrypoints."""

from scripts.main import *  # noqa: F401,F403
from scripts.main import main as _scripts_main

if __name__ == "__main__":
    _scripts_main()
