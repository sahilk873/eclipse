# ECLIPSE: Parameter Calibration and Literature Grounding

This document grounds the simulation parameters in `config/default.yaml` in ED and queueing literature. All parameters are configurable; values below represent plausible mid-volume ED settings.

## Arrival Rate (lambda)

- **Default:** 4.0 patients/hour
- **Rationale:** EDs with ~40,000 annual visits average roughly 4–6 patients per hour (Becker's Hospital Review, CDC NCHS). Lambda = 4 aligns with a mid-volume ED. Actual arrival rates vary by time of day and day of week.
- **References:** CDC Data Brief 503 (2022); Becker's Hospital Review median ED visit times by volume; PMC5052838 (ED visit volume variability).

## Service Times (lognormal by risk class)

- **Critical:** mean 30 min, std 15 min
- **Urgent:** mean 20 min, std 10 min
- **Low:** mean 15 min, std 8 min
- **Rationale:** Lognormal captures right-skewed service times. Critical patients typically require longer care; low-acuity patients have shorter service times. Values are consistent with ED length-of-stay studies and ESI-based acuity.
- **References:** ESI Implementation Handbook; ED length-of-stay literature (e.g., GWU HSRC duration of ED visits).

## Patience (exponential mean, minutes)

- **Critical:** 120 min
- **Urgent:** 60 min
- **Low:** 30 min
- **Rationale:** Reflects willingness to wait; high-acuity patients tend to wait longer before leaving. Exponential is standard for queueing models of abandonment.
- **References:** Queueing/ED abandonment literature.

## Risk Mix

- **Critical:** 10%, **Urgent:** 30%, **Low:** 60%
- **Rationale:** Typical ED acuity distribution; majority low-acuity, minority critical.
- **References:** ED acuity mix studies; ESI level distributions.

## Deterioration

- **critical_delay_min:** 15 min
- **hazard_base:** 0.01
- **hazard_per_min:** 0.002
- **Rationale:** Hazard increases with wait for high-risk patients. Captures time-to-critical deterioration.
- **References:** Time-to-critical deterioration and boarding literature.

## Constraints

- **missed_critical_epsilon:** 0.02 (2%)
- **critical_TTC_minutes:** 30
- **critical_TTC_exceed_pct:** 0.05 (5%)
- **Rationale:** Align with clinical targets for high-acuity patients: minimize missed critical cases and keep time-to-critical within acceptable bounds.
- **References:** ESI implementation; high-acuity patient benchmarks (e.g., triage within 15 min for Level 1–2).

## Episode Length (T)

- **Default:** 480 minutes (8 hours)
- **Rationale:** Represents a shift or half-day simulation window for evaluation.

## Queue and Servers

- **Qmax:** 20 (queue length threshold for overload)
- **n_servers:** 3
- **Rationale:** Configurable to represent staffing; Qmax for overload metrics.
