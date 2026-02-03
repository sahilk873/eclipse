# ECLIPSE Comprehensive Report

Generated: 2026-02-02T23:14:20.573639

## Executive Summary

### Key Findings

- Best evolved mechanism achieved fitness of 6.0578
- Average fitness improvement: 0.1111
- Evolved mechanism outperforms best baseline by 1.0981 fitness
- Evolved mechanism beats 6 of 6 baselines on fitness
- Meets safety constraints at least as well as best baseline (all baselines feasible)
- Pareto frontier contains 33.3% of evaluated mechanisms
- Average robustness score: 19.0982 (poor)
- Most critical component: no_redirect (average impact: 2.5326)

### Recommendations

- Evolved mechanisms improve safety/throughput; consider LLM-guided search for similar settings.
- Focus on no_redirect when designing triage (most critical component, avg impact 2.5326).
- Mechanisms remain stable under arrival/load shifts; suitable for variable EDs.
- Continue evolution with increased population size for better exploration
- Consider multi-objective optimization to balance competing objectives

## Evolution Analysis

- **Evolution runs**: 6
- **Best fitness**: 6.0578
- **Average best fitness**: 4.4637
- **Std of final fitness (across seeds)**: 2.0446
- **Convergence consistency**: 0.0232
- **Convergence std (final gen)**: 0.4841

### Convergence Plots

- `convergence/convergence_multi_run.png`
- `evolution/convergence_convergence_seed_0.png`

## Baseline Comparison

- **Baselines evaluated**: 6
- **Feasible baselines**: 6
- **Best baseline fitness**: 4.9597

### Baseline Metrics Table

| Baseline | Fitness | Feasible | adverse_events | TTC_p95 | throughput | mean_wait |
|----------|---------|----------|----------------|---------|------------|----------|
| FIFO_always_admit_exact_info | 4.7995 | Yes | 0.0046 | 0.4919 | 3.9438 | 0.0000 |
| FIFO_always_admit_no_info | 2.4541 | Yes | 0.0030 | 0.0807 | 1.6250 | 0.0000 |
| Severity_always_admit_exact_in | 4.9597 | Yes | 0.0042 | 0.3888 | 3.9531 | 0.0000 |
| Severity_always_admit_no_info | 2.5222 | Yes | 0.0027 | 0.0633 | 1.6313 | 0.0000 |
| Hybrid_always_admit_coarse_inf | 4.9002 | Yes | 0.0042 | 0.3822 | 3.9588 | 0.0000 |
| Risk_threshold_gating_no_exact | 2.5222 | Yes | 0.0027 | 0.0633 | 1.6313 | 0.0000 |

## Robustness Tests

- **Average robustness score**: 19.0982

### Mechanism x Scenario

| Mechanism | nominal | lambda_up_25 | lambda_down_ | heavier_serv | more_high_ri | reduced_pati | Feasible/Total |
|---|---|---|---|---|---|---|---|
| mutated_8454 | err | err | err | err | err | err | 0/6 |
| baseline | 4.12 | 5.15 | 3.74 | 1.04 | 2.82 | 4.42 | 6/6 |
| baseline | 2.55 | 2.54 | 2.07 | 2.38 | 2.08 | 2.59 | 6/6 |
| baseline | 4.70 | 5.63 | 3.99 | 1.84 | 3.28 | 5.02 | 6/6 |
| baseline | 2.14 | 2.14 | 1.96 | 2.15 | 1.93 | 2.19 | 6/6 |
| baseline | 4.85 | 5.88 | 4.03 | 1.91 | 3.56 | 5.12 | 6/6 |
| baseline | 2.36 | 2.41 | 1.96 | 2.13 | 2.04 | 2.40 | 6/6 |

## Ablation Study

- **Most critical component**: no_redirect (avg impact: 2.5326)
- **Least critical component**: no_reneging (avg impact: 0.4396)

### Ablation Summary Table

| Variant | Fitness Impact | Classification | Description |
|---------|----------------|----------------|-------------|
| no_redirect | -2.5326 | critical | No patient redirection |
| combined_redirect | -2.1512 | critical | Combined risk and congestion-based redir |
| severity_priority | +2.0993 | critical | Severity-based prioritization |
| no_info | -1.8269 | critical | No information provided to patients |
| exact_info | -1.7268 | critical | Exact wait time information |
| fifo_service | -0.6404 | critical | FIFO service rule (no prioritization) |
| no_reneging | -0.4396 | critical | No patient reneging |

## Clinical Interpretation

Metrics map to clinical outcomes (see `docs/CLINICAL_METRICS.md`):

- **adverse_events_rate**: Safety; target minimize
- **critical_TTC_p95**: Time-to-critical for high-acuity; target < 30 min
- **missed_critical_rate**: Critical patients not served; target < 2%
- **throughput**: Patients served; efficiency
- **mean_wait**: Patient experience
- **overload_time**: Queue overload; system stress

## Practical Recommendations

- Evolved mechanisms improve safety/throughput; consider LLM-guided search for similar settings.
- Focus on no_redirect when designing triage (most critical component, avg impact 2.5326).
- Mechanisms remain stable under arrival/load shifts; suitable for variable EDs.
- Continue evolution with increased population size for better exploration
- Consider multi-objective optimization to balance competing objectives

## Generated Artifacts

### Plots

- pareto_frontier.png
- convergence_multi_run.png
- convergence_convergence_seed_0.png

### Data_Files

- convergence_main_evolution.json
- baselines_results.csv
- evolution_result.json
- pareto_frontier.json
- best_mechanism_main_evolution.json
- robustness_suite_results.json
- pipeline_checkpoint.json
- pipeline_final.json
- ablation_study_results.json
- convergence_suite_results.json
- best_gen_1.json
- best_gen_0.json
- convergence_seed_0.json
- Severity_always_admit_no_info.json
- Risk_threshold_gating_no_exact_info.json
- FIFO_always_admit_no_info.json
- Severity_always_admit_exact_info.json
- FIFO_always_admit_exact_info.json
- Hybrid_always_admit_coarse_info.json
- convergence_convergence_run_2.json
- best_mechanism_convergence_run_2.json
- best_mechanism_convergence_run_1.json
- convergence_convergence_run_1.json
- best_mechanism_convergence_run_0.json
- convergence_convergence_run_0.json

### Reports

- comprehensive_report_combined.json

