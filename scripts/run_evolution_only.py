"""Run evolution only (no baselines/robustness/ablation)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    from eval.run_evolution import run_evolution
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_every", type=int, default=1)
    args = p.parse_args()
    run_evolution(config_path=args.config, master_seed=args.seed, save_every=args.save_every)


if __name__ == "__main__":
    main()
