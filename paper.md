---
title: 'ECLIPSE: Evolutionary Clinical Intake and Prioritization System'
authors:
  - name: ECLIPSE Contributors
    orcid: '0000-0000-0000-0000'
    affiliation: 1
affiliations:
  - name: Placeholder
    index: 1
date: 01 January 2025
bibliography: paper.bib
---

# Summary

ECLIPSE is an open-source Python framework for designing triage and intake mechanisms for simulated emergency department (ED) queues. It combines discrete-event simulation with evolutionary search and optional large language model (LLM)-guided mutation to discover mechanism designs that optimize clinically constrained objectives—prioritizing safety (adverse events, critical time-to-care) while balancing throughput and patient wait times.

# Statement of Need

Emergency department overcrowding and suboptimal triage policies are persistent challenges in healthcare operations [@hurwitz2014]. Existing approaches include fixed protocols (e.g., Emergency Severity Index, ESI), simulation-based evaluation of hand-crafted policies [@ashour2012], and neural-network metamodels for multiobjective optimization [@healthcare_mgmt_sci]. However, no open tool supports *evolutionary search over a composable mechanism space* with LLM-guided proposal of novel variants. ECLIPSE fills this gap, enabling researchers and practitioners to explore mechanism designs (info disclosure, service priority rules, redirect policies) without manual enumeration.

# State of the Field

| Approach | Limitation |
|----------|------------|
| FAHP-MAUT triage [@ashour2012] | Single fixed algorithm |
| ESI | Static protocol |
| DES frameworks [@hurwitz2014] | Evaluate given policies, no mechanism search |
| NN metamodels | Fixed mechanism design |

ECLIPSE contributes: (1) LLM-guided evolutionary mechanism search; (2) composable mechanism space (info policy, service rule, redirect modes); (3) multi-objective design with Pareto frontier and safety constraints; (4) adaptive mutation and stagnation handling. See `docs/POSITIONING.md` for detailed comparison.

# Software Design

- **Simulation** (`sim/`): Discrete-event ED queue with risk classes, deterioration, patience, and configurable parameters.
- **Mechanisms** (`mechanisms/`): JSON-serializable genome with info policy, service rule, redirect policy; validation and mutation operators.
- **Evolution** (`evolution/`): Population-based search with elitism, adaptive mutation strength, optional LLM mutation.
- **Evaluation** (`eval/`): Episodes, baselines, robustness under distribution shifts, ablation studies.
- **Reporting** (`scripts/make_report.py`): Baseline comparison, convergence analysis, robustness tables, ablation component importance.

Design trade-offs: LLM mutation requires an API key and adds latency; we support random-only evolution as a fallback. Parameters are grounded in literature (`docs/CALIBRATION.md`).

# Research Impact Statement

- **Benchmarks**: Evolved mechanisms are evaluated against seven baselines (FIFO, severity priority, hybrid, ESI-like gating, and an FAHP-MAUT–inspired baseline [@ashour2012]) with reported fitness, feasibility, and key metrics.
- **Convergence and robustness**: Multi-seed convergence suite; robustness under λ ±25%, heavier service tail, more high-risk mix, reduced patience.
- **Ablation**: Component importance analysis identifies which mechanism parts (info disclosure, redirect, service rule) matter most.
- **Reproducibility**: Fixed seeds, `reproducibility_info.json`, single-command benchmark script.

# Limitations

- **Variance across seeds**: Evolution exhibits high variance; we report mean ± 95% CI across multiple seeds. Consider increasing seeds (≥5) for stronger statistical claims.
- **Simulation fidelity**: Parameters are literature-calibrated but idealized; real-world EDs vary in arrival patterns, acuity mix, and staffing.
- **Opacity criterion**: Partial disclosure (coarse_bins/none) may not emerge in all runs; the report includes opacity_emergence_rate for transparency.
- **Computational cost**: Full pipeline (30 generations, 5 convergence runs, robustness, ablations) can take hours; reduce generations or population for quick runs.

# AI Usage Disclosure

LLM-guided mutation (optional) uses the OpenAI API to propose mechanism variants. When enabled, the LLM receives fitness feedback and constraint violations to inform proposals. All LLM outputs are validated and normalized before evaluation. Core simulation, evolution loop, and reporting logic are human-authored.

# References

- Ashour & Kremer. A simulation analysis of the impact of FAHP-MAUT triage algorithm. *Expert Systems with Applications* (2012).
- Hurwitz et al. A flexible simulation platform to quantify and manage ED crowding. *BMC Medical Informatics and Decision Making* (2014).
