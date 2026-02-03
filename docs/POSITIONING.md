# ECLIPSE: Positioning vs. Prior Work

This document compares ECLIPSE to existing approaches in emergency department (ED) triage optimization and simulation, and states ECLIPSE's unique contributions.

## Prior Work Comparison

| Approach | Description | Key Limitation |
|----------|-------------|----------------|
| **FAHP-MAUT triage** (Ashour & Kremer) | Fuzzy Analytic Hierarchy Process + Multi-Attribute Utility Theory for triage algorithm design; compared to ESI via simulation | Single fixed algorithm; no mechanism space exploration |
| **ESI (Emergency Severity Index)** | 5-level standard triage tool; widely deployed clinically | Static protocol; does not adapt to site-specific conditions |
| **DES frameworks** (e.g., Hurwitz et al.) | Discrete-event simulation to quantify ED crowding; flexible platforms for managerial decisions | Evaluate fixed policies; no automated mechanism search |
| **Neural-network metamodels** (Health Care Mgmt Sci) | Simulation-based multiobjective optimization with NN metamodels for low-acuity fast-track; Pareto solutions | Fixed mechanism design; metamodel for evaluation, not for mechanism generation |
| **ML + simulation resource scheduling** | Machine learning combined with simulation for resource scheduling and patient flow | Focus on resource allocation; not on triage mechanism design |
| **Queue theory / load-leveling** | Queue-theoretic metrics for ED resource planning | Analytical; limited mechanism design space |

## ECLIPSE Contributions

1. **LLM-guided evolutionary mechanism search**  
   Uses large language models to propose mechanism variants informed by fitness feedback and constraint violations. No prior ED triage work combines LLMs with evolutionary search for mechanism design.

2. **Mechanism space exploration**  
   Searches over composable mechanism components (info policy: none/coarse_bins/exact; service rule: fifo/severity_priority/hybrid; redirect modes; reneging). Existing work typically evaluates fixed algorithms or hand-crafted variants.

3. **Multi-objective design with safety constraints**  
   Pareto frontier over safety, throughput, and wait; hard constraints (missed_critical_rate, critical_TTC). Balances competing objectives while enforcing clinical safety.

4. **Adaptive mutation and stagnation handling**  
   Mutation strength adapts to convergence stagnation; LLM fraction configurable. Improves exploration when the population stagnates.

## Build vs. Contribute Justification

- **Why not extend existing simulators?**  
  Standard ED simulators (e.g., Simio, Arena, custom DES) evaluate given policies. They do not support evolutionary search over a mechanism space or LLM-guided proposal. ECLIPSE integrates evolution and evaluation in one framework.

- **Why not contribute to existing triage optimization packages?**  
  There is no open-source package that combines (a) discrete-event ED simulation, (b) evolutionary mechanism search, and (c) LLM-guided mutation. ECLIPSE fills this gap.

- **Why a new tool?**  
  The combination of LLM + evolution + clinical constraints for triage mechanism design is novel. Existing tools do not support this workflow.

## References (Representative)

- Ashour, O.M., Kremer, G.E.O. A simulation analysis of the impact of FAHP–MAUT triage algorithm on the Emergency Department performance measures. *Expert Systems with Applications* 39(12):10472–10483 (2012). https://doi.org/10.1016/j.eswa.2012.02.163
- Hurwitz, J.E. et al. A flexible simulation platform to quantify and manage emergency department crowding. *BMC Medical Informatics and Decision Making* 14:50 (2014). https://doi.org/10.1186/1472-6947-14-50
- ESI Implementation Handbook. Emergency Severity Index, Version 4. AHRQ. https://www.ahrq.gov/patient-safety/settings/emergency-dept/esi.html
- Health Care Management Science: multiobjective simulation-optimization, low-acuity fast-track (2024). Springer.
