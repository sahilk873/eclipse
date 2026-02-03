#!/usr/bin/env python3
"""
Run the full ECLIPSE pipeline for reproducible benchmarking.

Executes: baselines, evolution, convergence suite, robustness, ablations,
Pareto analysis, and comprehensive report. Uses fixed seed for deterministic
results.

Usage:
  python scripts/run_reproducible_benchmark.py --seed 0
  python scripts/run_reproducible_benchmark.py --seed 42 --results_dir results/run_42
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Run reproducible ECLIPSE benchmark (full pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_reproducible_benchmark.py --seed 0
  python scripts/run_reproducible_benchmark.py --seed 42 --results_dir results/run_42
  python scripts/run_reproducible_benchmark.py --config pipeline_config.json
        """,
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (default: 0)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration JSON file (default: uses config/default.yaml via pipeline)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory (default: results)",
    )
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM mutation")

    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_complete_pipeline.py"),
        "--seed",
        str(args.seed),
        "--results",
        args.results_dir,
        "--full-pipeline",
    ]
    if args.config:
        cmd.extend(["--config", args.config])
    if args.use_llm:
        cmd.append("--use-llm")

    print(f"Running: {' '.join(cmd)}")
    print(f"Seed: {args.seed}, Results: {args.results_dir}")
    sys.exit(subprocess.call(cmd, cwd=str(PROJECT_ROOT)))


if __name__ == "__main__":
    main()
