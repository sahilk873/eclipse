"""Run full benchmark: baselines, evolution (short), robustness, ablation, plots."""

from __future__ import annotations

import sys
from pathlib import Path

# Run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main() -> None:
    from eval.run_baselines import main as run_baselines
    from eval.run_evolution import run_evolution
    from eval.run_robustness import main as run_robustness
    from eval.run_ablation import main as run_ablation
    from eval.plots import generate_all_plots

    results_dir = Path(__file__).resolve().parent.parent / "results"
    config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"

    print("1. Running baselines (n_episodes=20 for quick run)...")
    run_baselines(n_episodes=20, config_path=config_path, results_dir=str(results_dir), base_seed=0)

    print("2. Running evolution (G=3, N=6, K=3 for quick run)...")
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["evolution"] = cfg.get("evolution", {}) | {"N": 6, "M": 3, "K": 3, "G": 3}
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as t:
        yaml.dump(cfg, t)
        tmp_config = t.name
    best_per_gen, _, _, _ = run_evolution(config_path=tmp_config, master_seed=0, save_every=1)
    Path(tmp_config).unlink(missing_ok=True)
    best_mechanism = best_per_gen[-1] if best_per_gen else {}
    (results_dir / "evolution").mkdir(parents=True, exist_ok=True)
    import json
    with open(results_dir / "evolution" / "best_quick.json", "w") as f:
        json.dump(best_mechanism, f, indent=2)

    print("3. Running robustness...")
    run_robustness(
        config_path=config_path,
        results_dir=str(results_dir),
        n_episodes=10,
        base_seed=0,
        evolved_path=results_dir / "evolution" / "best_quick.json",
    )

    print("4. Running ablation...")
    run_ablation(
        mechanism_path=results_dir / "evolution" / "best_quick.json",
        config_path=config_path,
        results_dir=str(results_dir),
        n_episodes=10,
        base_seed=0,
    )

    print("5. Generating plots...")
    generate_all_plots(results_dir=str(results_dir))
    print("Done.")


if __name__ == "__main__":
    main()
