"""CLI entry point for testing an exported policy interactively in Genesis."""
from __future__ import annotations

import argparse
from pathlib import Path

from microduck_genesis.infer import run_interactive


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONNX policy interactively in Genesis World")
    parser.add_argument("policy", type=Path, help="Path to exported policy.onnx")
    parser.add_argument("--headless", action="store_true", help="Run without graphical viewer")

    args = parser.parse_args()
    if not args.policy.exists():
        raise SystemExit(f"Policy file not found: {args.policy}")

    run_interactive(args.policy, headless=args.headless)


if __name__ == "__main__":
    main()
