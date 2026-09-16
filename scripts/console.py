#!/usr/bin/env python3
"""Convenience runner for web console and camera streamer."""
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microduck_genesis.cli.console_cli import main

if __name__ == "__main__":
    main()
