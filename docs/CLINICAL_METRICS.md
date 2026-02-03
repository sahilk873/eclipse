# ECLIPSE: Clinical Metrics

This document maps ECLIPSE evaluation metrics to clinical outcomes and operational targets.

## Metrics

| Metric | Clinical Meaning | Target |
|--------|------------------|--------|
| **adverse_events_rate** | Safety; rate of adverse events (e.g., deterioration, harm) | Minimize |
| **critical_TTC_p95** | 95th percentile time-to-critical for high-acuity patients | < 30 min |
| **missed_critical_rate** | Proportion of critical patients not served (redirected or left) | < 2% |
| **throughput** | Patients served; operational efficiency | Maximize |
| **mean_wait** | Average wait time; patient experience | Minimize |
| **overload_time** | Time the queue exceeds capacity; system stress | Minimize |

## Fitness Weights (config/default.yaml)

- **A (adverse_events):** 100 — heavily penalize adverse events
- **B (critical_TTC_p95):** 1.0 — penalize long time-to-critical
- **C (overload_time):** 0.1 — penalize overload
- **D (throughput):** 2.0 — reward throughput
- **E (mean_wait):** 0.5 — penalize long waits

## Constraints

- **missed_critical_epsilon:** Max 2% of critical patients missed
- **critical_TTC_minutes:** Target time-to-critical 30 min
- **critical_TTC_exceed_pct:** Max 5% of critical patients exceeding T_crit

A mechanism is **feasible** only if all constraints are satisfied.
