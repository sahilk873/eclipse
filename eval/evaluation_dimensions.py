"""
Evaluation dimensions for ECLIPSE: generalization, external validation, robustness,
clinical plausibility, failure modes, and transparency.

These dimensions are first-class in the codebase: the pipeline and report
explicitly address each. See docs/EVALUATION_DIMENSIONS.md for definitions
and mapping to artifacts.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Canonical list of evaluation dimensions (single source of truth)
EVALUATION_DIMENSIONS = [
    "generalization",
    "external_validation",
    "robustness",
    "clinical_plausibility",
    "failure_modes",
    "transparency",
]

# How each dimension is addressed in code and artifacts
DIMENSION_SOURCES = {
    "generalization": {
        "description": "Performance across seeds and settings; not overfit to a single run.",
        "code_artifacts": [
            "scripts/run_convergence_suite.py",
            "results/convergence/convergence_suite_results.json",
            "convergence_multi_run.png",
        ],
        "report_keys": ["convergence_analysis", "num_evolution_runs", "fitness_ci_95_lower", "fitness_ci_95_upper"],
    },
    "external_validation": {
        "description": "Comparison against fixed baselines (FIFO, severity, hybrid, ESI-like) not used in search.",
        "code_artifacts": [
            "eval/run_baselines.py",
            "scripts/make_report.py (baseline_comparison)",
            "results/baselines_results.csv",
        ],
        "report_keys": ["baseline_comparison", "baselines_beaten", "statistical_comparison"],
    },
    "robustness": {
        "description": "Stability under distribution shift (arrival rate ±25%, service tail, acuity mix, patience).",
        "code_artifacts": [
            "scripts/run_robustness_suite.py",
            "eval/run_robustness.py",
            "results/robustness/robustness_suite_results.json",
        ],
        "report_keys": ["robustness_analysis", "robustness_summary", "per_mechanism_robustness"],
    },
    "clinical_plausibility": {
        "description": "Metrics and constraints grounded in clinical targets (ESI, time-to-critical, missed critical).",
        "code_artifacts": [
            "docs/CALIBRATION.md",
            "docs/CLINICAL_METRICS.md",
            "config/default.yaml (constraints, fitness)",
            "eval/metrics.py (check_constraints, compute_fitness)",
        ],
        "report_keys": ["clinical_interpretation", "constraints", "feasible_baselines"],
    },
    "failure_modes": {
        "description": "Explicit tracking of constraint violations and adverse outcomes; LLM receives failure analysis.",
        "code_artifacts": [
            "evolution/llm_mutation.py (failure_analysis_from_metrics, parent_failures)",
            "evolution/evolve.py (parent_failures)",
            "eval/metrics.py (check_constraints, constraints_violated)",
            "scripts/run_ablation_suite.py",
        ],
        "report_keys": ["ablation_analysis", "constraints_violated", "failure analysis in LLM prompt"],
    },
    "transparency": {
        "description": "Mechanisms are interpretable JSON; reports and artifacts are human-inspectable.",
        "code_artifacts": [
            "mechanisms/schema.py",
            "mechanisms/models.py",
            "results/*.json (mechanisms, reports)",
            "scripts/make_report.py (markdown + JSON report)",
            "opacity_analysis (info_mode emergence)",
        ],
        "report_keys": ["opacity_analysis", "artifacts", "mechanism JSON schema"],
    },
}


def get_dimension_evidence(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Map the comprehensive report to evidence per evaluation dimension.
    Used by make_report to populate the evaluation dimensions section.
    """
    evidence: Dict[str, Dict[str, Any]] = {}
    for dim in EVALUATION_DIMENSIONS:
        meta = DIMENSION_SOURCES[dim]
        evidence[dim] = {
            "description": meta["description"],
            "addressed": False,
            "summary": None,
            "report_refs": [],
        }
        for key in meta["report_keys"]:
            if key in report and report[key]:
                evidence[dim]["addressed"] = True
                evidence[dim]["report_refs"].append(key)
        # Dimension-specific summary from report
        if dim == "generalization" and report.get("convergence_analysis"):
            c = report["convergence_analysis"]
            n = c.get("num_evolution_runs", 0)
            if n >= 2:
                evidence[dim]["summary"] = (
                    f"Multi-run convergence: {n} runs; "
                    f"fitness 95% CI [{c.get('fitness_ci_95_lower', 0):.2f}, {c.get('fitness_ci_95_upper', 0):.2f}]"
                )
        elif dim == "external_validation" and report.get("baseline_comparison"):
            b = report["baseline_comparison"]
            evidence[dim]["summary"] = (
                f"Evolved vs {b.get('num_baselines', 0)} baselines; "
                f"best baseline fitness {b.get('best_baseline_fitness', 0):.2f}"
            )
            conv = report.get("convergence_analysis")
            baseline_table = b.get("baseline_table", [])
            evolved_fit = conv.get("best_overall_fitness") if conv else None
            if baseline_table and evolved_fit is not None:
                beats = sum(1 for row in baseline_table if evolved_fit > row.get("fitness", 0))
                evidence[dim]["summary"] += f"; evolved beats {beats} of {len(baseline_table)} baselines"
        elif dim == "robustness" and report.get("robustness_analysis"):
            r = report["robustness_analysis"]
            rs = r.get("robustness_summary", {})
            evidence[dim]["summary"] = (
                f"Shifts: max degradation {rs.get('max_degradation_percent', 0):.1f}%; "
                f"feasible in {rs.get('feasible_scenarios_count', 0)}/{rs.get('total_scenarios', 6)} scenarios"
            )
        elif dim == "clinical_plausibility":
            evidence[dim]["addressed"] = True
            evidence[dim]["summary"] = (
                "Constraints: missed_critical <2%, TTC target 30 min; "
                "fitness weights (A–E) and metrics from docs/CLINICAL_METRICS.md"
            )
        elif dim == "failure_modes":
            if report.get("ablation_analysis"):
                a = report["ablation_analysis"]
                evidence[dim]["addressed"] = True
                if a.get("most_important"):
                    name, impact = a["most_important"]
                    evidence[dim]["summary"] = f"Ablation: most critical component {name} (impact {impact:.4f}); constraint violations tracked"
                else:
                    evidence[dim]["summary"] = "Ablation and constraint violations tracked; LLM receives failure analysis"
            else:
                evidence[dim]["summary"] = "Constraint violations and failure analysis fed to LLM mutation"
                evidence[dim]["addressed"] = True
        elif dim == "transparency":
            evidence[dim]["addressed"] = True
            evidence[dim]["summary"] = "Mechanisms as JSON; markdown + JSON report; opacity (info_mode) analysis"
            if report.get("opacity_analysis"):
                op = report["opacity_analysis"]
                evidence[dim]["summary"] += f"; coarse_bins/none in {op.get('opacity_emergence_rate', 0)*100:.0f}% of runs"
    return evidence


def evaluation_dimensions_report_section(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the evaluation dimensions section for the comprehensive report.
    """
    evidence = get_dimension_evidence(report)
    return {
        "dimensions": EVALUATION_DIMENSIONS,
        "evidence": evidence,
        "doc_ref": "docs/EVALUATION_DIMENSIONS.md",
    }
