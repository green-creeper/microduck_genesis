"""Compatibility shim for `duck-sim`: forwards to microduck_genesis body_cli."""
import sys
from pathlib import Path

# Ensure src is on sys.path
root_src = Path(__file__).resolve().parents[2]
if str(root_src) not in sys.path:
    sys.path.insert(0, str(root_src))

from microduck_genesis.cli.body_cli import main

if __name__ == "__main__":
    main()
