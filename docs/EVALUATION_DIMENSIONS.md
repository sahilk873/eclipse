# ECLIPSE: Evaluation Dimensions

ECLIPSE treats the following six dimensions as first-class evaluation concerns. The pipeline and comprehensive report explicitly address each; this document defines them and points to where they are implemented in the code.

| Dimension | Definition | Where in code / artifacts |
|-----------|-------------|---------------------------|
| **Generalization** | Performance across seeds and settings; mechanisms should not be overfit to a single run. | Multi-seed convergence suite (`scripts/run_convergence_suite.py`), `convergence_suite_results.json`, convergence plots; report: `convergence_analysis`, fitness 95% CI. |
| **External validation** | Comparison against fixed baselines (FIFO, severity, hybrid, ESI-like) that are *not* used in the search. | `eval/run_baselines.py`, `results/baselines_results.csv`; report: `baseline_comparison`, `statistical_comparison`, baselines beaten. |
| **Robustness** | Stability under distribution shift (arrival rate ±25%, heavier service tail, different acuity mix, reduced patience). | `scripts/run_robustness_suite.py`, `eval/run_robustness.py`, `results/robustness/robustness_suite_results.json`; report: `robustness_analysis`, max degradation, feasible scenarios. |
| **Clinical plausibility** | Metrics and constraints grounded in clinical targets (e.g. ESI, time-to-critical, missed critical rate). | `docs/CALIBRATION.md`, `docs/CLINICAL_METRICS.md`; `config/default.yaml` (constraints, fitness weights); `eval/metrics.py` (constraints, fitness). |
| **Failure modes** | Explicit tracking of constraint violations and adverse outcomes; failure analysis fed into LLM to steer proposals. | `evolution/llm_mutation.py` (`failure_analysis_from_metrics`, parent_failures in prompt); `evolution/evolve.py` (parent_failures); `eval/metrics.py` (check_constraints); ablation suite for component-level failure impact. |
| **Transparency** | Mechanisms are interpretable (JSON schema); reports and artifacts are human-inspectable; disclosure (info_mode) analysis. | `mechanisms/schema.py`, `mechanisms/models.py`; JSON/Markdown reports; `opacity_analysis` (coarse_bins/none emergence). |

## Report section

The comprehensive report (`comprehensive_report_combined.json` / `.md`) includes an **Evaluation dimensions** section that summarizes, for each dimension, whether evidence was produced in the run and a short summary (e.g. robustness degradation, baselines beaten). The logic lives in `eval/evaluation_dimensions.py` and is used by `scripts/make_report.py`.

## Pipeline coverage

- **Generalization**: Enable `pipeline.run_convergence_suite: true` and run the full pipeline (multiple seeds).
- **External validation**: Baselines are always run; report compares evolved vs baselines.
- **Robustness**: Enable `pipeline.run_robustness: true` to evaluate under shifts.
- **Clinical plausibility**: Always applied via config and `eval/metrics.py`.
- **Failure modes**: Constraint violations and failure bullets are always tracked; ablation adds component-level evidence when `pipeline.run_ablations: true`.
- **Transparency**: Mechanism JSON and reports are always produced; opacity analysis is in the report when convergence/evolution results exist.
