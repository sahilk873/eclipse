# ECLIPSE

**Evolutionary Clinical Intake and Prioritization System**

LLM-guided evolutionary search for designing triage and intake mechanisms for a simulated clinic/ED queue, optimizing a clinically constrained objective (safety first), with benchmarking against standard heuristic and queueing baselines.

## One sentence

Build an LLM-guided evolutionary search system that designs triage and intake mechanisms for a simulated clinic/ED queue, optimizing a clinically constrained objective (safety first), and benchmark against standard heuristic and queueing baselines using Pareto dominance, convergence, and robustness tests.

## Requirements

- Python 3.11+
- Dependencies: `numpy`, `pandas`, `matplotlib`, `pyyaml`, `jsonschema`
- Optional (for LLM mutation): `openai` and `OPENAI_API_KEY`

```bash
pip install -r requirements.txt
```

## Project layout

```
ECLIPSE/
├── config/default.yaml    # Experiment config (sim, constraints, fitness, evolution)
├── docs/                   # POSITIONING.md, CALIBRATION.md, CLINICAL_METRICS.md
├── sim/                    # Discrete-event simulator
├── mechanisms/            # Mechanism schema, validation, mutation
├── evolution/              # Selection, reproduction, LLM mutation
├── baselines/              # Six baseline mechanisms
├── eval/                   # run_episodes, run_baselines, run_evolution, run_robustness, run_ablation, plots
├── results/                # Output CSVs, JSONs, figures
├── scripts/                # CLI entrypoints, make_report, run_reproducible_benchmark
├── LICENSE                 # MIT
├── CITATION.cff            # Citation metadata
├── pyproject.toml          # Python package metadata
├── paper.md                # JOSS paper (summary, statement of need, state of field)
└── CONTRIBUTING.md         # Contribution guidelines
```

## Related work / Positioning

See [docs/POSITIONING.md](docs/POSITIONING.md) for comparison to FAHP-MAUT triage, ESI, DES frameworks, neural-network metamodels, and ECLIPSE's contributions (LLM-guided evolution, mechanism search, multi-objective design).

**Additional documentation:**
- [docs/CALIBRATION.md](docs/CALIBRATION.md) — Parameter grounding in ED literature
- [docs/CLINICAL_METRICS.md](docs/CLINICAL_METRICS.md) — Metrics and clinical outcomes

## Reproducibility

- **Seeds**: All scripts accept `--seed` (or `--config`). The simulator and evolution use deterministic RNG given the seed.
- **Config**: `config/default.yaml` defines T, λ, servers, service/deterioration/patience params, fitness weights (A–E), constraint bounds, and evolution N/M/K/G.
- **Saving**: Every evaluated mechanism can be saved as JSON; metrics are written to CSV. Figures can be regenerated from saved CSVs/JSONs without re-running evolution.

### Reproducing results

To run the full pipeline (baselines, evolution, convergence, robustness, ablations, Pareto, report) with a fixed seed:

```bash
python scripts/run_reproducible_benchmark.py --seed 0
```

With LLM mutation (requires `OPENAI_API_KEY`):

```bash
python scripts/run_reproducible_benchmark.py --seed 0 --use-llm
```

**Expected outputs:** `results/baselines_results.csv`, `results/evolution_result.json`, `results/convergence/`, `results/robustness/`, `results/ablations/`, `results/pareto_frontier.json`, `results/comprehensive_report_combined.md`, `results/reproducibility_info.json`.

**Determinism:** A fixed seed yields deterministic results (given the same Python version and dependencies). `reproducibility_info.json` records seed, config hash, timestamp, and git commit for traceability.

## How to run

### 1. Run baselines (200 episodes each)

```bash
cd ECLIPSE
python -m eval.run_baselines --n_episodes 200 --results_dir results --seed 0
```

Output: `results/baselines_results.csv`, `results/baselines/<name>.json`.

### 2. Run evolution (no LLM)

```bash
python -m eval.run_evolution --config config/default.yaml --seed 0 --save_every 1
```

Output: `results/evolution/best_gen_<g>.json`, `results/evolution/convergence_seed_0.json`.

### 3. Run evolution with LLM mutation (optional)

**By default LLM mutation is off** (`evolution.use_llm: false` in `config/default.yaml`). Evolution runs using only random mutation, so no API key is required.

To use LLM-guided mutation:

1. Set `evolution.use_llm: true` in `config/default.yaml` (or pass a config with that set).
2. Install the OpenAI client: `pip install openai`.
3. Set the `OPENAI_API_KEY` environment variable (or pass it into the LLM mutation layer).
4. Run evolution as above. The loop will use the LLM to propose 1–3 new mechanisms per generation (about half of offspring from LLM, half from random mutation). If the import fails or no API key is set, the code falls back to random mutation only.

LLM mutation uses the **OpenAI Responses API** with **Structured Outputs** when available; otherwise it falls back to Chat Completions. Set `OPENAI_LLM_MODEL` in `.env` (default: `gpt-4o-mini`). Use `gpt-5-mini` or `gpt-4o` for better JSON. Automatically retries without temperature for models that reject it (e.g. gpt-5-mini) and falls back to `gpt-4o-mini` on model errors.

### 4. Robustness (re-evaluate under shifts)

```bash
python -m eval.run_robustness --results_dir results --n_episodes 200 --evolved results/evolution/best_gen_29.json
```

Output: `results/robustness_results.csv`.

### 5. Ablation (flip one component of best mechanism)

```bash
python -m eval.run_ablation results/evolution/best_gen_29.json --results_dir results --n_episodes 200
```

Output: `results/ablation_results.csv`.

### 6. Generate plots

```bash
python -m eval.plots --results_dir results
```

Generates: `results/pareto_frontier.png`, `results/evolution/convergence_*.png`, `results/robustness_bars.png`, `results/ablation_bars.png`.

## Mechanism schema (genome)

Each mechanism is a JSON object with:

- **Info**: `info_mode` (none | coarse_bins | exact); optional `info_bins`, `risk_labeling`
- **Gating**: `gating_mode` (always_admit | threshold | probabilistic | capacity_based); conditional `gating_threshold`, `capacity_load`
- **Priority**: `service_rule` (fifo | severity_priority | hybrid); if hybrid, `hybrid_a`, `hybrid_b`
- **Redirect/exit**: `redirect_low_risk`, `redirect_threshold`, `reneging_enabled`, `reneging_model`

Validation: `mechanisms.schema.validate_mechanism(mechanism)`.

## Baselines

1. FIFO + always admit + exact wait info  
2. FIFO + always admit + no wait info  
3. Severity priority + always admit + exact info  
4. Severity priority + always admit + no info  
5. Hybrid priority (fixed weights) + always admit + coarse info  
6. Risk-threshold gating (ESI-like): redirect low risk when load high, no exact wait info  

## Acceptance criteria

- Evolved mechanisms beat at least 3 strong baselines on Pareto metrics.  
- Safety constraints satisfied at least as well as baselines.  
- Opacity or partial disclosure (e.g. coarse_bins or none) emerges in multiple seeds.  
- Performance remains strong under at least 2 robustness shifts (λ ±25%, heavier service tail, more high-risk, or stricter patience).

## Limitations

- High variance across seeds; report uses mean ± 95% CI when multiple runs available.
- Simulation parameters are idealized; real EDs vary.
- Full pipeline (30 gen, 5 seeds, robustness, ablations) is computationally intensive.

## Citing

If you use ECLIPSE in your research, please cite it. A `CITATION.cff` file is included; you can also use:

```bibtex
@software{eclipse,
  title = {ECLIPSE: Evolutionary Clinical Intake and Prioritization System},
  year = {2025},
  license = {MIT},
}
```

## License

MIT License. See [LICENSE](LICENSE).
