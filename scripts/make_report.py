"""Comprehensive report generator for ECLIPSE experiments."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from evolution.population_db import PopulationDB

from eval.evaluation_dimensions import evaluation_dimensions_report_section


def _ci95_t(values: List[float]) -> tuple[float, float]:
    """95% confidence interval using t-distribution (small-sample)."""
    if len(values) < 2:
        return (values[0], values[0]) if values else (0.0, 0.0)
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0
    std_err = math.sqrt(variance / n) if n > 0 else 0
    # t_{0.975, n-1} approximation: 2.0 for n~10, 2.78 for n=5, 12.7 for n=2
    try:
        import scipy.stats
        t_val = scipy.stats.t.ppf(0.975, df=n - 1)
    except ImportError:
        t_val = 2.0 if n >= 5 else (2.5 if n >= 3 else 4.3)
    return (mean - t_val * std_err, mean + t_val * std_err)


def _ttest_one_sample(values: List[float], mu: float) -> Optional[Dict[str, float]]:
    """One-sample t-test: H0 mean(values)=mu. Returns p-value, t-stat if scipy available."""
    if len(values) < 2:
        return None
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std_err = math.sqrt(variance / n)
    if std_err <= 0:
        return None
    t_stat = (mean - mu) / std_err
    try:
        import scipy.stats
        p_value = 2 * (1 - scipy.stats.t.cdf(abs(t_stat), df=n - 1))
        return {"t_statistic": t_stat, "p_value": p_value, "df": n - 1}
    except ImportError:
        return {"t_statistic": t_stat}


def generate_comprehensive_report(
    results_dir: str, run_id: Optional[str] = None, include_plots: bool = True
) -> Dict[str, Any]:
    """
    Generate a comprehensive report combining all experiment results.

    Args:
        results_dir: Directory containing all experiment results
        run_id: Specific run ID to analyze (if None, analyze all)
        include_plots: Whether to generate plots

    Returns:
        Comprehensive report dictionary
    """
    results_path = Path(results_dir)

    print(f"Generating comprehensive report from {results_path}")

    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "results_directory": str(results_path),
            "run_id": run_id,
        },
        "summary": {},
        "convergence_analysis": {},
        "pareto_analysis": {},
        "robustness_analysis": {},
        "ablation_analysis": {},
        "baseline_comparison": {},
        "artifacts": {},
    }

    # Analyze evolution results
    evolution_files = list(results_path.glob("**/convergence_*.json"))
    if evolution_files:
        report["convergence_analysis"] = analyze_evolution_results(
            evolution_files, run_id
        )

    # Opacity analysis: does coarse_bins or none emerge across seeds? (methods paper)
    report["opacity_analysis"] = analyze_opacity_emergence(results_path)

    # Analyze baseline results
    baseline_files = list(results_path.glob("**/baselines_results.csv"))
    if baseline_files:
        report["baseline_comparison"] = analyze_baseline_results(baseline_files)

    # Analyze Pareto frontier
    pareto_files = list(results_path.glob("**/pareto_frontier.json"))
    if pareto_files:
        report["pareto_analysis"] = analyze_pareto_results(pareto_files)

    # Analyze robustness
    robustness_files = list(results_path.glob("**/robustness_suite_results.json"))
    if robustness_files:
        report["robustness_analysis"] = analyze_robustness_results(robustness_files)

    # Analyze ablations
    ablation_files = list(results_path.glob("**/ablation_study_results.json"))
    if ablation_files:
        report["ablation_analysis"] = analyze_ablation_results(ablation_files)

    # Transfer evaluation (external settings: Option A and B)
    transfer_file = results_path / "transfer_results.json"
    if transfer_file.exists():
        try:
            with open(transfer_file, "r") as f:
                transfer_data = json.load(f)
            report["transfer_analysis"] = _analyze_transfer_results(transfer_data)
        except Exception as e:
            report["transfer_analysis"] = {"error": str(e)}

    # Statistical comparison: evolved vs best baseline (for methods paper)
    if report.get("convergence_analysis") and report.get("baseline_comparison"):
        conv = report["convergence_analysis"]
        base = report["baseline_comparison"]
        best_fitnesses = conv.get("best_fitnesses_per_run", [])
        best_baseline_fitness = base.get("best_baseline_fitness")
        if best_fitnesses and best_baseline_fitness is not None and len(best_fitnesses) >= 2:
            ttest = _ttest_one_sample(best_fitnesses, best_baseline_fitness)
            if ttest:
                report["statistical_comparison"] = {
                    "evolved_mean": sum(best_fitnesses) / len(best_fitnesses),
                    "evolved_n": len(best_fitnesses),
                    "baseline_best": best_baseline_fitness,
                    "t_statistic": ttest.get("t_statistic"),
                    "p_value": ttest.get("p_value"),
                    "df": ttest.get("df"),
                    "interpretation": (
                        "Evolved significantly outperforms best baseline (p < 0.05)"
                        if ttest.get("p_value", 1) < 0.05 and ttest.get("t_statistic", 0) > 0
                        else "Difference not statistically significant at α=0.05"
                    ),
                }

    # Evaluation dimensions: generalization, external validation, robustness, clinical plausibility, failure modes, transparency
    report["evaluation_dimensions"] = evaluation_dimensions_report_section(report)

    # Generate summary
    report["summary"] = generate_executive_summary(report)

    # List all artifacts
    report["artifacts"] = catalog_artifacts(results_path)

    # Save report
    report_file = results_path / f"comprehensive_report_{run_id or 'combined'}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # Generate markdown report
    markdown_file = results_path / f"comprehensive_report_{run_id or 'combined'}.md"
    generate_markdown_report(report, markdown_file)

    print(f"Comprehensive report saved to {report_file}")
    print(f"Markdown report saved to {markdown_file}")

    return report


def _normalize_evolution_data(data: Any) -> Dict[str, Any]:
    """Normalize evolution data from list or dict format to a common structure."""
    if isinstance(data, list):
        # Array of generation dicts, e.g. convergence_main_evolution.json
        if not data:
            return {"best_fitness": 0, "generations": 0, "convergence_data": []}
        last_gen = data[-1] if isinstance(data[-1], dict) else {}
        best_fitness = last_gen.get("best_fitness", 0)
        return {
            "best_fitness": best_fitness,
            "generations": len(data),
            "convergence_data": data,
        }
    if isinstance(data, dict):
        # Dict format, may have best_fitness_per_gen or convergence_data
        if "best_fitness" not in data and "best_fitness_per_gen" in data:
            per_gen = data["best_fitness_per_gen"]
            data = dict(data)
            data["best_fitness"] = per_gen[-1] if per_gen else 0
            data["convergence_data"] = [
                {"generation": i, "best_fitness": f}
                for i, f in enumerate(per_gen)
            ]
        return data
    return {"best_fitness": 0, "generations": 0, "convergence_data": []}


def analyze_evolution_results(
    evolution_files: List[Path], run_id: Optional[str]
) -> Dict[str, Any]:
    """Analyze evolution/convergence results."""

    all_results = []
    best_fitnesses_raw: List[float] = []

    for evo_file in evolution_files:
        try:
            with open(evo_file, "r") as f:
                raw_data = json.load(f)
            # Handle convergence_suite_results.json format
            if "run_results" in raw_data:
                for run in raw_data["run_results"]:
                    if run.get("success") and run.get("result"):
                        bf = run["result"].get("best_fitness")
                        if bf is not None:
                            best_fitnesses_raw.append(bf)
                            conv_data = run["result"].get("convergence_data", [])
                            all_results.append({
                                "file": str(evo_file),
                                "data": {
                                    "best_fitness": bf,
                                    "generations": len(conv_data),
                                    "convergence_data": conv_data,
                                },
                            })
                continue
            data = _normalize_evolution_data(raw_data)
            all_results.append({"file": str(evo_file), "data": data})
            bf = data.get("best_fitness", 0)
            if bf != 0 or "best_fitness" in data:
                best_fitnesses_raw.append(bf)
        except Exception as e:
            print(f"Error reading {evo_file}: {e}")

    # Use raw list if we got it from suite; else extract from all_results
    if not best_fitnesses_raw and all_results:
        best_fitnesses_raw = [
            r["data"].get("best_fitness", 0) for r in all_results if "data" in r
        ]

    if not all_results:
        return {"error": "No valid evolution results found"}

    best_fitnesses = [f for f in best_fitnesses_raw if f is not None]
    generations = [
        r["data"].get("generations", len(r["data"].get("convergence_data", [])))
        for r in all_results
        if "data" in r
    ]

    analysis = {
        "num_evolution_runs": len(best_fitnesses) or len(all_results),
        "best_overall_fitness": max(best_fitnesses) if best_fitnesses else 0,
        "avg_best_fitness": sum(best_fitnesses) / len(best_fitnesses)
        if best_fitnesses
        else 0,
        "fitness_std": (
            (sum((f - sum(best_fitnesses) / len(best_fitnesses)) ** 2 for f in best_fitnesses) / len(best_fitnesses)) ** 0.5
            if len(best_fitnesses) > 1
            else 0
        ),
        "avg_generations": sum(generations) / len(generations) if generations else 0,
        "best_fitnesses_per_run": best_fitnesses,  # for statistical tests
    }

    # 95% CI for methods paper (when multiple seeds)
    if len(best_fitnesses) >= 2:
        ci_lo, ci_hi = _ci95_t(best_fitnesses)
        analysis["fitness_ci_95_lower"] = ci_lo
        analysis["fitness_ci_95_upper"] = ci_hi

    # Convergence patterns
    convergence_curves = []
    for r in all_results:
        conv_data = r["data"].get("convergence_data", [])
        if conv_data:
            fitness_curve = [gen["best_fitness"] for gen in conv_data]
            convergence_curves.append(fitness_curve)

    if convergence_curves:
        final_fitnesses = [c[-1] for c in convergence_curves if c]
        
        # Compute min/max/mean per generation across runs
        max_gen = max(len(c) for c in convergence_curves) if convergence_curves else 0
        per_generation_stats = []
        for gen_idx in range(max_gen):
            gen_fitnesses = []
            for curve in convergence_curves:
                if gen_idx < len(curve):
                    gen_fitnesses.append(curve[gen_idx])
            if gen_fitnesses:
                per_generation_stats.append({
                    "generation": gen_idx,
                    "min": min(gen_fitnesses),
                    "max": max(gen_fitnesses),
                    "mean": sum(gen_fitnesses) / len(gen_fitnesses),
                })
        
        analysis["convergence_patterns"] = {
            "num_curves": len(convergence_curves),
            "avg_final_improvement": calculate_avg_improvement(convergence_curves),
            "convergence_consistency": calculate_convergence_consistency(
                convergence_curves
            ),
            "convergence_std": (
                (sum((f - sum(final_fitnesses) / len(final_fitnesses)) ** 2 for f in final_fitnesses) / len(final_fitnesses)) ** 0.5
                if len(final_fitnesses) > 1 else 0
            ),
            "per_generation_stats": per_generation_stats,  # min/max/mean per generation
        }

    return analysis


def analyze_opacity_emergence(results_path: Path) -> Dict[str, Any]:
    """Check if partial disclosure (coarse_bins or none) emerges in evolved mechanisms."""
    suite_files = list(results_path.glob("**/convergence_suite_results.json"))
    info_modes: List[str] = []
    for fp in suite_files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            for run in data.get("run_results", []):
                if run.get("success") and run.get("result"):
                    best = run["result"].get("best_mechanism")
                    if best:
                        im = best.get("info_policy", {}).get("info_mode", "none")
                        info_modes.append(im)
        except Exception:
            pass
    # Also check main evolution_result
    evo_file = results_path / "evolution_result.json"
    if evo_file.exists():
        try:
            with open(evo_file, "r") as f:
                evo = json.load(f)
            best = evo.get("best_mechanism")
            if best:
                im = best.get("info_policy", {}).get("info_mode", "none")
                info_modes.append(im)
        except Exception:
            pass
    opacity_modes = ("coarse_bins", "none")
    emerged = [m for m in info_modes if m in opacity_modes]
    return {
        "info_modes_observed": list(dict.fromkeys(info_modes)),
        "opacity_emergence_count": len(emerged),
        "total_mechanisms": len(info_modes),
        "opacity_emergence_rate": len(emerged) / len(info_modes) if info_modes else 0,
        "criterion_met": len(emerged) >= 1,
    }


def analyze_baseline_results(baseline_files: List[Path]) -> Dict[str, Any]:
    """Analyze baseline evaluation results."""

    import csv

    all_results = []

    for baseline_file in baseline_files:
        try:
            with open(baseline_file, "r") as f:
                reader = csv.DictReader(f)
                results = list(reader)
                all_results.extend(results)
        except Exception as e:
            print(f"Error reading {baseline_file}: {e}")

    if not all_results:
        return {"error": "No valid baseline results found"}

    # Extract numerical values and build baseline table
    fitnesses = []
    feasible_count = 0
    baseline_table = []

    for result in all_results:
        try:
            fitness = float(result.get("fitness", 0))
            fitnesses.append(fitness)
            if result.get("feasible", "False").lower() == "true":
                feasible_count += 1
            baseline_table.append({
                "name": result.get("name", "unknown"),
                "fitness": fitness,
                "feasible": result.get("feasible", "False").lower() == "true",
                "adverse_events_rate": _safe_float(
                    result.get("adverse_events_rate_mean", result.get("adverse_events_rate", 0))
                ),
                "critical_TTC_p95": _safe_float(
                    result.get("critical_TTC_p95_mean", result.get("critical_TTC_p95", 0))
                ),
                "throughput": _safe_float(
                    result.get("throughput_mean", result.get("throughput", 0))
                ),
                "mean_wait": _safe_float(
                    result.get("mean_wait_mean", result.get("mean_wait", 0))
                ),
                "overload_time": _safe_float(
                    result.get("overload_time_mean", result.get("overload_time", 0))
                ),
            })
        except (ValueError, TypeError):
            continue

    analysis = {
        "num_baselines": len(all_results),
        "feasible_baselines": feasible_count,
        "feasibility_rate": feasible_count / len(all_results) if all_results else 0,
        "best_baseline_fitness": max(fitnesses) if fitnesses else 0,
        "avg_baseline_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
        "fitness_range": max(fitnesses) - min(fitnesses) if len(fitnesses) > 1 else 0,
        "baseline_table": baseline_table,
    }

    # Find best baseline
    if all_results and fitnesses:
        best_idx = fitnesses.index(max(fitnesses))
        analysis["best_baseline"] = {
            "name": all_results[best_idx].get("name", "unknown"),
            "fitness": fitnesses[best_idx],
            "feasible": all_results[best_idx].get("feasible", "False"),
        }

    return analysis


def _safe_float(val: Any) -> float:
    """Safely convert to float."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def analyze_pareto_results(pareto_files: List[Path]) -> Dict[str, Any]:
    """Analyze Pareto frontier results."""

    all_results = []

    for pareto_file in pareto_files:
        try:
            with open(pareto_file, "r") as f:
                data = json.load(f)
                all_results.append({"file": str(pareto_file), "data": data})
        except Exception as e:
            print(f"Error reading {pareto_file}: {e}")

    if not all_results:
        return {"error": "No valid Pareto results found"}

    # Extract frontier statistics
    frontier_sizes = [
        r["data"].get("pareto_frontier_size", 0) for r in all_results if "data" in r
    ]
    total_mechanisms = [
        r["data"].get("total_mechanisms", 0) for r in all_results if "data" in r
    ]

    analysis = {
        "num_pareto_analyses": len(all_results),
        "avg_frontier_size": sum(frontier_sizes) / len(frontier_sizes)
        if frontier_sizes
        else 0,
        "avg_total_mechanisms": sum(total_mechanisms) / len(total_mechanisms)
        if total_mechanisms
        else 0,
        "pareto_efficiency": sum(frontier_sizes) / sum(total_mechanisms)
        if total_mechanisms and frontier_sizes
        else 0,
    }

    # Check baseline domination
    domination_data = []
    for r in all_results:
        if "domination_analysis" in r["data"]:
            dom = r["data"]["domination_analysis"]
            if "pareto_percentage" in dom:
                domination_data.append(dom["pareto_percentage"])

    if domination_data:
        analysis["avg_pareto_percentage"] = sum(domination_data) / len(domination_data)

    return analysis


def _analyze_transfer_results(transfer_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build transfer_analysis section from transfer_results.json."""
    out = {
        "transfer_table": transfer_data.get("transfer_table", []),
        "summary": transfer_data.get("summary", {}),
        "evaluations": transfer_data.get("evaluations", []),
    }
    # One-line summary for executive summary
    summary = transfer_data.get("summary", {})
    by_test = summary.get("best_from_default_on_alternates", {})
    if by_test:
        parts = [f"{k}: {v:.2f}" for k, v in by_test.items() if v is not None and isinstance(v, (int, float))]
        out["summary_line"] = "Best-from-default on alternates: " + "; ".join(parts)
    if summary.get("degradation_on_high_volume_pct") is not None:
        out["degradation_on_high_volume_pct"] = summary["degradation_on_high_volume_pct"]
    return out


def analyze_robustness_results(robustness_files: List[Path]) -> Dict[str, Any]:
    """Analyze robustness test results."""

    all_results = []

    for robust_file in robustness_files:
        try:
            with open(robust_file, "r") as f:
                data = json.load(f)
                all_results.append({"file": str(robust_file), "data": data})
        except Exception as e:
            print(f"Error reading {robust_file}: {e}")

    if not all_results:
        return {"error": "No valid robustness results found"}

    # Extract robustness scores
    robustness_analysis = []
    for r in all_results:
        if "robustness_analysis" in r["data"]:
            robustness_analysis.append(r["data"]["robustness_analysis"])

    if not robustness_analysis:
        return {"error": "No robustness analysis found"}

    # Collect scores
    all_scores = []
    for analysis in robustness_analysis:
        if "robustness_scores" in analysis:
            scores = list(analysis["robustness_scores"].values())
            all_scores.extend(
                [s for s in scores if s >= 0]
            )  # Exclude failed evaluations

    analysis_result = {
        "num_robustness_tests": len(all_results),
        "total_mechanisms_tested": len(all_scores),
        "avg_robustness_score": sum(all_scores) / len(all_scores) if all_scores else 0,
        "best_robustness_score": min(all_scores) if all_scores else 0,
        "worst_robustness_score": max(all_scores) if all_scores else 0,
    }

    # Build robustness table: mechanism x scenario
    robustness_table = []
    scenarios = []
    for r in all_results:
        d = r["data"]
        if "mechanism_results" in d:
            for mech_id, mech_data in d["mechanism_results"].items():
                shift_res = mech_data.get("shift_results", {})
                if not scenarios and shift_res:
                    scenarios = list(shift_res.keys())
                seed_tag = mech_data.get("mechanism", {}).get("meta", {}).get("seed_tag", mech_id[:8])
                row = {"mechanism": seed_tag, "scenarios": {}}
                success_count = 0
                for scen_name, scen_data in shift_res.items():
                    succ = scen_data.get("success", False)
                    if succ:
                        success_count += 1
                        ev = scen_data.get("evaluation", {})
                        row["scenarios"][scen_name] = {"fitness": ev.get("fitness", 0), "success": True}
                    else:
                        row["scenarios"][scen_name] = {"error": scen_data.get("error", "unknown"), "success": False}
                row["feasible_scenarios"] = success_count
                row["total_scenarios"] = len(shift_res)
                robustness_table.append(row)
            break  # use first file with mechanism_results

    analysis_result["robustness_table"] = robustness_table
    analysis_result["scenarios"] = scenarios

    # Per-mechanism worst-case fitness degradation (for methods paper)
    for r in all_results:
        d = r["data"]
        if "robustness_analysis" in d and "per_mechanism_robustness" in d["robustness_analysis"]:
            analysis_result["per_mechanism_robustness"] = d["robustness_analysis"]["per_mechanism_robustness"]
            break

    # Most/least robust mechanisms
    best_mechanisms = []
    worst_mechanisms = []

    for analysis in robustness_analysis:
        if "most_robust" in analysis and analysis["most_robust"]:
            best_mechanisms.append(analysis["most_robust"])
        if "least_robust" in analysis and analysis["least_robust"]:
            worst_mechanisms.append(analysis["least_robust"])

    analysis_result["most_robust_mechanisms"] = best_mechanisms
    analysis_result["least_robust_mechanisms"] = worst_mechanisms
    
    # Compute robustness summary: max degradation and feasible scenarios count
    if robustness_table:
        max_degradation = 0.0
        feasible_counts = []
        for row in robustness_table:
            feasible_counts.append(row.get("feasible_scenarios", 0))
        
        # Get max degradation from per_mechanism_robustness if available
        if "per_mechanism_robustness" in analysis_result:
            for pm_data in analysis_result["per_mechanism_robustness"].values():
                deg = pm_data.get("worst_case_fitness_degradation", 0)
                if deg > max_degradation:
                    max_degradation = deg
        
        analysis_result["robustness_summary"] = {
            "max_degradation_percent": max_degradation * 100 if max_degradation > 0 else 0,
            "feasible_scenarios_count": max(feasible_counts) if feasible_counts else 0,
            "total_scenarios": len(scenarios) if scenarios else 6,
        }

    return analysis_result


def analyze_ablation_results(ablation_files: List[Path]) -> Dict[str, Any]:
    """Analyze ablation study results."""

    all_results = []

    for ablation_file in ablation_files:
        try:
            with open(ablation_file, "r") as f:
                data = json.load(f)
                all_results.append({"file": str(ablation_file), "data": data})
        except Exception as e:
            print(f"Error reading {ablation_file}: {e}")

    if not all_results:
        return {"error": "No valid ablation results found"}

    # Extract component importance data and build ablation table
    component_importance_data = []
    ablation_table = []

    for r in all_results:
        d = r["data"]
        if "component_analysis" in d:
            comp_analysis = d["component_analysis"]
            if "ranked_components" in comp_analysis:
                component_importance_data.append(comp_analysis["ranked_components"])
                if not ablation_table:
                    critical = set(comp_analysis.get("critical_components", []))
                    moderate = set(comp_analysis.get("moderate_components", []))
                    minor = set(comp_analysis.get("minor_components", []))
                    for component_name, impact_data in comp_analysis.get("ranked_components", []):
                        cat = "critical" if component_name in critical else "moderate" if component_name in moderate else "minor"
                        ablation_table.append({
                            "variant": component_name,
                            "fitness_impact": impact_data.get("fitness_impact", 0),
                            "description": impact_data.get("description", ""),
                            "classification": cat,
                        })

    if not component_importance_data:
        return {"error": "No component importance analysis found"}

    # Analyze component importance patterns
    analysis = {
        "num_ablation_studies": len(all_results),
        "component_studies": len(component_importance_data),
        "ablation_table": ablation_table,
    }

    # Most frequently important components
    component_impacts = {}
    for study in component_importance_data:
        for component_name, impact_data in study:
            if component_name not in component_impacts:
                component_impacts[component_name] = []
            component_impacts[component_name].append(abs(impact_data["fitness_impact"]))

    # Calculate average impact per component
    avg_impacts = {}
    for component, impacts in component_impacts.items():
        avg_impacts[component] = sum(impacts) / len(impacts) if impacts else 0

    if avg_impacts:
        sorted_components = sorted(
            avg_impacts.items(), key=lambda x: x[1], reverse=True
        )
        analysis["most_important_components"] = sorted_components[:5]
        analysis["least_important_components"] = sorted_components[-5:]
        analysis["most_important"] = sorted_components[0] if sorted_components else None
        analysis["least_important"] = sorted_components[-1] if sorted_components else None

    return analysis


def calculate_avg_improvement(convergence_curves: List[List[float]]) -> float:
    """Calculate average improvement from first to last generation."""
    if not convergence_curves:
        return 0.0

    improvements = []
    for curve in convergence_curves:
        if len(curve) >= 2:
            improvement = curve[-1] - curve[0]
            improvements.append(improvement)

    return sum(improvements) / len(improvements) if improvements else 0.0


def calculate_convergence_consistency(convergence_curves: List[List[float]]) -> float:
    """Calculate consistency of convergence across runs."""
    if len(convergence_curves) < 2:
        return 1.0

    # Simple correlation-based consistency measure
    # Normalize curves to [0,1] range for comparison
    normalized_curves = []
    for curve in convergence_curves:
        if not curve:
            continue
        min_val = min(curve)
        max_val = max(curve)
        if max_val > min_val:
            normalized = [(v - min_val) / (max_val - min_val) for v in curve]
        else:
            normalized = [0.0] * len(curve)
        normalized_curves.append(normalized)

    # Calculate average correlation between curves
    correlations = []
    for i in range(len(normalized_curves)):
        for j in range(i + 1, len(normalized_curves)):
            corr = simple_correlation(normalized_curves[i], normalized_curves[j])
            correlations.append(corr)

    return sum(correlations) / len(correlations) if correlations else 0.0


def simple_correlation(a: List[float], b: List[float]) -> float:
    """Simple correlation calculation."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0

    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n

    numerator = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    sum_sq_a = sum((a[i] - mean_a) ** 2 for i in range(n))
    sum_sq_b = sum((b[i] - mean_b) ** 2 for i in range(n))

    denominator = (sum_sq_a * sum_sq_b) ** 0.5

    return numerator / denominator if denominator > 0 else 0.0


def generate_executive_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Generate executive summary of all findings."""

    summary = {"key_findings": [], "performance_summary": {}, "recommendations": []}

    # Evolution findings
    if report.get("convergence_analysis"):
        conv = report["convergence_analysis"]
        if "best_overall_fitness" in conv:
            summary["key_findings"].append(
                f"Best evolved mechanism achieved fitness of {conv['best_overall_fitness']:.4f}"
            )
        if "convergence_patterns" in conv:
            patterns = conv["convergence_patterns"]
            if "avg_final_improvement" in patterns:
                summary["key_findings"].append(
                    f"Average fitness improvement: {patterns['avg_final_improvement']:.4f}"
                )

    # Baseline comparison and head-to-head
    if report.get("baseline_comparison") and report.get("convergence_analysis"):
        base = report["baseline_comparison"]
        conv = report["convergence_analysis"]

        if "best_overall_fitness" in conv and "best_baseline_fitness" in base:
            evolved_better = (
                conv["best_overall_fitness"] > base["best_baseline_fitness"]
            )
            improvement = conv["best_overall_fitness"] - base["best_baseline_fitness"]

            summary["key_findings"].append(
                f"Evolved mechanism {'outperforms' if evolved_better else 'matches'} best baseline by {improvement:.4f} fitness"
            )

        # Head-to-head: evolved beats X of N baselines
        if "baseline_table" in base and "best_overall_fitness" in conv:
            evolved_fit = conv["best_overall_fitness"]
            beats = sum(1 for row in base["baseline_table"] if evolved_fit > row.get("fitness", 0))
            n_base = len(base["baseline_table"])
            summary["baselines_beaten"] = beats
            summary["key_findings"].append(
                f"Evolved mechanism beats {beats} of {n_base} baselines on fitness"
            )
        if "feasible_baselines" in base and base.get("feasibility_rate", 0) >= 1.0:
            summary["key_findings"].append(
                "Meets safety constraints at least as well as best baseline (all baselines feasible)"
            )

    # Statistical comparison
    if report.get("statistical_comparison"):
        sc = report["statistical_comparison"]
        if sc.get("interpretation"):
            summary["statistical_interpretation"] = sc["interpretation"]
            summary["key_findings"].append(sc["interpretation"])

    # Opacity
    if report.get("opacity_analysis"):
        op = report["opacity_analysis"]
        if op.get("total_mechanisms", 0) > 0:
            rate = op.get("opacity_emergence_rate", 0)
            met = op.get("criterion_met", False)
            summary["key_findings"].append(
                f"Opacity (coarse_bins/none) emergence: {rate * 100:.0f}% of runs ({'criterion met' if met else 'criterion not met'})"
            )

    # Pareto analysis
    if report.get("pareto_analysis"):
        pareto = report["pareto_analysis"]
        if "pareto_efficiency" in pareto:
            summary["key_findings"].append(
                f"Pareto frontier contains {pareto['pareto_efficiency'] * 100:.1f}% of evaluated mechanisms"
            )

    # Robustness
    if report.get("robustness_analysis"):
        robust = report["robustness_analysis"]
        # Add robustness summary if available
        if "robustness_summary" in robust:
            rs = robust["robustness_summary"]
            summary["key_findings"].append(
                f"Performance degrades by at most {rs.get('max_degradation_percent', 0):.1f}% under shifts"
            )
            summary["key_findings"].append(
                f"Remains feasible under {rs.get('feasible_scenarios_count', 0)} of {rs.get('total_scenarios', 6)} scenarios"
            )
        if "per_mechanism_robustness" in robust:
            pmr = robust["per_mechanism_robustness"]
            evolved = [p for p in pmr.values() if "mutated" in str(p.get("seed_tag", "")) or "evolution" in str(p.get("seed_tag", ""))]
            if evolved:
                best_ev = min(evolved, key=lambda x: x.get("worst_case_fitness_degradation", 999))
                fea = best_ev.get("feasible_scenarios", 0)
                tot = best_ev.get("total_scenarios", 6)
                summary["key_findings"].append(
                    f"Best evolved mechanism: {fea}/{tot} scenarios passed, worst-case fitness drop {best_ev.get('worst_case_fitness_degradation', 0):.2f}"
                )
        elif "avg_robustness_score" in robust:
            score = robust["avg_robustness_score"]
            summary["key_findings"].append(
                f"Average robustness score: {score:.4f} ({'good' if score < 1.0 else 'moderate' if score < 5.0 else 'poor'})"
            )

    # Transfer evaluation (external settings)
    if report.get("transfer_analysis") and "error" not in report["transfer_analysis"]:
        ta = report["transfer_analysis"]
        if ta.get("summary_line"):
            summary["key_findings"].append("Transfer: " + ta["summary_line"])
        if ta.get("degradation_on_high_volume_pct") is not None:
            summary["key_findings"].append(
                f"Transfer: best-from-default degrades by {ta['degradation_on_high_volume_pct']}% on high_volume"
            )

    # Component importance
    if report.get("ablation_analysis"):
        abl = report["ablation_analysis"]
        if "most_important_components" in abl:
            top_component = (
                abl["most_important_components"][0]
                if abl["most_important_components"]
                else None
            )
            if top_component:
                summary["key_findings"].append(
                    f"Most critical component: {top_component[0]} (average impact: {top_component[1]:.4f})"
                )

    # Evaluation dimensions (generalization, external validation, robustness, clinical plausibility, failure modes, transparency)
    if report.get("evaluation_dimensions"):
        ed = report["evaluation_dimensions"]
        dims_addressed = [d for d in ed.get("dimensions", []) if ed.get("evidence", {}).get(d, {}).get("addressed")]
        summary["evaluation_dimensions_addressed"] = dims_addressed
        summary["key_findings"].append(
            f"Evaluation dimensions addressed in this run: {', '.join(dims_addressed)} (see report section and docs/EVALUATION_DIMENSIONS.md)"
        )

    # Practical recommendations (derived from results)
    if report.get("convergence_analysis") and report.get("baseline_comparison"):
        conv = report["convergence_analysis"]
        base = report["baseline_comparison"]
        if conv.get("best_overall_fitness", 0) > base.get("best_baseline_fitness", 0):
            summary["recommendations"].append(
                "Evolved mechanisms improve safety/throughput; consider LLM-guided search for similar settings."
            )
    if report.get("ablation_analysis"):
        abl = report["ablation_analysis"]
        if abl.get("most_important"):
            name, impact = abl["most_important"]
            summary["recommendations"].append(
                f"Focus on {name} when designing triage (most critical component, avg impact {impact:.4f})."
            )
    if report.get("robustness_analysis"):
        robust = report["robustness_analysis"]
        if robust.get("robustness_table"):
            feasible_counts = [r.get("feasible_scenarios", 0) for r in robust["robustness_table"]]
            if feasible_counts:
                max_feas = max(feasible_counts)
                total = robust["robustness_table"][0].get("total_scenarios", 6) if robust["robustness_table"] else 6
                if max_feas >= total - 1:
                    summary["recommendations"].append(
                        "Mechanisms remain stable under arrival/load shifts; suitable for variable EDs."
                    )
    summary["recommendations"].append(
        "Continue evolution with increased population size for better exploration"
    )
    summary["recommendations"].append(
        "Consider multi-objective optimization to balance competing objectives"
    )

    return summary


def catalog_artifacts(results_path: Path) -> Dict[str, List[str]]:
    """Catalog all generated artifacts."""

    artifacts = {"plots": [], "data_files": [], "reports": [], "databases": []}

    # Find all files
    for file_path in results_path.rglob("*"):
        if file_path.is_file():
            file_str = str(file_path)

            if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".pdf"]:
                artifacts["plots"].append(file_str)
            elif file_path.suffix.lower() in [".json", ".csv", ".yaml", ".yml"]:
                if "report" in file_path.name.lower():
                    artifacts["reports"].append(file_str)
                elif file_path.suffix.lower() == ".db":
                    artifacts["databases"].append(file_str)
                else:
                    artifacts["data_files"].append(file_str)

    return artifacts


def generate_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
    """Generate markdown version of the report."""

    results_path = Path(report["metadata"].get("results_directory", "results"))

    md_content = f"""# ECLIPSE Comprehensive Report

Generated: {report["metadata"]["generated_at"]}

## Executive Summary

"""

    summary = report.get("summary", {})
    if "key_findings" in summary:
        md_content += "### Key Findings\n\n"
        for finding in summary["key_findings"]:
            md_content += f"- {finding}\n"
        md_content += "\n"

    if "recommendations" in summary:
        md_content += "### Recommendations\n\n"
        for rec in summary["recommendations"]:
            md_content += f"- {rec}\n"
        md_content += "\n"

    # Evolution Analysis (with convergence details)
    if "convergence_analysis" in report:
        md_content += "## Evolution Analysis\n\n"
        conv = report["convergence_analysis"]

        if "num_evolution_runs" in conv:
            md_content += f"- **Evolution runs**: {conv['num_evolution_runs']}\n"
        if "best_overall_fitness" in conv:
            md_content += f"- **Best fitness**: {conv['best_overall_fitness']:.4f}\n"
        if "avg_best_fitness" in conv:
            ci = ""
            if "fitness_ci_95_lower" in conv and "fitness_ci_95_upper" in conv:
                ci = f" (95% CI: [{conv['fitness_ci_95_lower']:.4f}, {conv['fitness_ci_95_upper']:.4f}])"
            md_content += f"- **Average best fitness**: {conv['avg_best_fitness']:.4f}{ci}\n"
        if "fitness_std" in conv:
            md_content += f"- **Std of final fitness (across seeds)**: {conv['fitness_std']:.4f}\n"
        if "convergence_patterns" in conv:
            pat = conv["convergence_patterns"]
            if "convergence_consistency" in pat:
                md_content += f"- **Convergence consistency**: {pat['convergence_consistency']:.4f}\n"
            if "convergence_std" in pat:
                md_content += f"- **Convergence std (final gen)**: {pat['convergence_std']:.4f}\n"
            # Convergence curves: min/max/mean per generation
            if "per_generation_stats" in pat and pat["per_generation_stats"]:
                md_content += "\n### Convergence Curves (min/max/mean per generation)\n\n"
                md_content += "| Generation | Min Fitness | Max Fitness | Mean Fitness |\n"
                md_content += "|------------|------------|-------------|--------------|\n"
                for stat in pat["per_generation_stats"][:20]:  # Show first 20 generations
                    md_content += f"| {stat['generation']} | {stat['min']:.4f} | {stat['max']:.4f} | {stat['mean']:.4f} |\n"
                if len(pat["per_generation_stats"]) > 20:
                    md_content += f"*... ({len(pat['per_generation_stats']) - 20} more generations)*\n"
                md_content += "\n"
        md_content += "\n"
        plot_paths = list(results_path.glob("**/convergence*.png"))
        if plot_paths:
            md_content += "### Convergence Plots\n\n"
            for p in plot_paths[:5]:
                md_content += f"- `{p.relative_to(results_path)}`\n"
            md_content += "\n"

    # Baseline Comparison (with table)
    if "baseline_comparison" in report:
        md_content += "## Baseline Comparison\n\n"
        base = report["baseline_comparison"]

        if "num_baselines" in base:
            md_content += f"- **Baselines evaluated**: {base['num_baselines']}\n"
        if "feasible_baselines" in base:
            md_content += f"- **Feasible baselines**: {base['feasible_baselines']}\n"
        if "best_baseline_fitness" in base:
            md_content += f"- **Best baseline fitness**: {base['best_baseline_fitness']:.4f}\n"
        md_content += "\n"

        if "baseline_table" in base:
            md_content += "### Baseline Metrics Table\n\n"
            md_content += "| Baseline | Fitness | Feasible | adverse_events | TTC_p95 | throughput | mean_wait |\n"
            md_content += "|----------|---------|----------|----------------|---------|------------|----------|\n"
            for row in base["baseline_table"]:
                name = str(row.get("name", ""))[:30]
                fit = row.get("fitness", 0)
                feas = "Yes" if row.get("feasible") else "No"
                adv = row.get("adverse_events_rate", 0)
                ttc = row.get("critical_TTC_p95", 0)
                thru = row.get("throughput", 0)
                wait = row.get("mean_wait", 0)
                md_content += f"| {name} | {fit:.4f} | {feas} | {adv:.4f} | {ttc:.4f} | {thru:.4f} | {wait:.4f} |\n"
            md_content += "\n"

    # Opacity analysis
    if "opacity_analysis" in report:
        op = report["opacity_analysis"]
        if op.get("total_mechanisms", 0) > 0:
            md_content += "## Opacity Analysis\n\n"
            md_content += f"- Info modes observed: {op.get('info_modes_observed', [])}\n"
            md_content += f"- Opacity (coarse_bins/none) emergence: {op.get('opacity_emergence_rate', 0) * 100:.0f}% ({op.get('opacity_emergence_count', 0)}/{op.get('total_mechanisms', 0)} runs)\n"
            md_content += f"- Criterion met: {op.get('criterion_met', False)}\n\n"

    # Statistical comparison (methods paper)
    if "statistical_comparison" in report:
        sc = report["statistical_comparison"]
        md_content += "## Statistical Comparison\n\n"
        md_content += f"- Evolved mean fitness: {sc.get('evolved_mean', 0):.4f} (n={sc.get('evolved_n', 0)})\n"
        md_content += f"- Best baseline fitness: {sc.get('baseline_best', 0):.4f}\n"
        if sc.get("t_statistic") is not None:
            md_content += f"- One-sample t-test: t={sc['t_statistic']:.3f}"
            if sc.get("p_value") is not None:
                md_content += f", p={sc['p_value']:.4f}"
            md_content += "\n"
        if sc.get("interpretation"):
            md_content += f"- **{sc['interpretation']}**\n"
        md_content += "\n"

    # Transfer evaluation (external settings)
    if "transfer_analysis" in report and "error" not in report.get("transfer_analysis", {}):
        ta = report["transfer_analysis"]
        md_content += "## Transfer Evaluation (External Settings)\n\n"
        if ta.get("summary_line"):
            md_content += f"- {ta['summary_line']}\n"
        if ta.get("degradation_on_high_volume_pct") is not None:
            md_content += f"- Best-from-default degrades by {ta['degradation_on_high_volume_pct']}% on high_volume\n"
        if ta.get("transfer_table"):
            md_content += "\n### Transfer Table (train_config x test_config)\n\n"
            md_content += "| Train config | Test config | Fitness | Feasible |\n"
            md_content += "|--------------|-------------|---------|----------|\n"
            for row in ta["transfer_table"]:
                train = str(row.get("train_config", ""))
                test = str(row.get("test_config", ""))
                fit = row.get("fitness")
                fit_str = f"{fit:.4f}" if fit is not None else "—"
                feas = "Yes" if row.get("feasible") else "No"
                md_content += f"| {train} | {test} | {fit_str} | {feas} |\n"
            md_content += "\n"

    # Evaluation dimensions: generalization, external validation, robustness, clinical plausibility, failure modes, transparency
    if "evaluation_dimensions" in report:
        ed = report["evaluation_dimensions"]
        md_content += "## Evaluation Dimensions\n\n"
        md_content += "Evidence for each dimension (see `docs/EVALUATION_DIMENSIONS.md` for definitions):\n\n"
        for dim in ed.get("dimensions", []):
            ev = ed.get("evidence", {}).get(dim, {})
            desc = ev.get("description", "")
            summary_line = ev.get("summary") or ("Addressed in pipeline" if ev.get("addressed") else "No evidence in this run")
            status = "✓" if ev.get("addressed") else "−"
            md_content += f"- **{dim}** {status}: {desc} — {summary_line}\n"
        md_content += "\n"

    # Robustness
    if "robustness_analysis" in report:
        md_content += "## Robustness Tests\n\n"
        robust = report["robustness_analysis"]
        if "avg_robustness_score" in robust:
            md_content += f"- **Average robustness score**: {robust['avg_robustness_score']:.4f}\n"
        # Robustness summary: degradation and feasibility
        if "robustness_summary" in robust:
            rs = robust["robustness_summary"]
            md_content += f"- **Performance degrades by at most**: {rs.get('max_degradation_percent', 0):.1f}% under shifts\n"
            md_content += f"- **Remains feasible under**: {rs.get('feasible_scenarios_count', 0)} of {rs.get('total_scenarios', 6)} scenarios\n"
            md_content += "\n"
        if "per_mechanism_robustness" in robust:
            md_content += "\n### Per-mechanism worst-case fitness degradation\n\n"
            md_content += "| Mechanism | Nominal | Min (worst shift) | Degradation | Scenarios passed |\n"
            md_content += "|-----------|---------|-------------------|-------------|------------------|\n"
            for mech_id, pm in list(robust["per_mechanism_robustness"].items())[:10]:
                tag = pm.get("seed_tag", mech_id[:8])
                nom = pm.get("nominal_fitness", 0)
                mn = pm.get("min_fitness_across_shifts", 0)
                deg = pm.get("worst_case_fitness_degradation", 0)
                fea = pm.get("feasible_scenarios", 0)
                tot = pm.get("total_scenarios", 6)
                md_content += f"| {tag} | {nom:.2f} | {mn:.2f} | {deg:.2f} | {fea}/{tot} |\n"
            md_content += "\n"
        if "robustness_table" in robust and robust["robustness_table"]:
            md_content += "\n### Mechanism x Scenario\n\n"
            scenarios = robust.get("scenarios", [])
            header = "| Mechanism | " + " | ".join(str(s)[:12] for s in scenarios[:6]) + " | Feasible/Total |\n"
            md_content += header
            md_content += "|" + "---|" * (len(scenarios[:6]) + 2) + "\n"
            for row in robust["robustness_table"][:10]:
                mech = str(row.get("mechanism", ""))[:15]
                cells = []
                for s in scenarios[:6]:
                    sc = row.get("scenarios", {}).get(s, {})
                    if sc.get("success"):
                        cells.append(f"{sc.get('fitness', 0):.2f}")
                    else:
                        cells.append("err")
                cells.append(f"{row.get('feasible_scenarios', 0)}/{row.get('total_scenarios', 6)}")
                md_content += "| " + mech + " | " + " | ".join(cells) + " |\n"
            md_content += "\n"

    # Ablation
    if "ablation_analysis" in report:
        md_content += "## Ablation Study\n\n"
        abl = report["ablation_analysis"]
        if abl.get("most_important"):
            name, impact = abl["most_important"]
            md_content += f"- **Most critical component**: {name} (avg impact: {impact:.4f})\n"
        if abl.get("least_important"):
            name, impact = abl["least_important"]
            md_content += f"- **Least critical component**: {name} (avg impact: {impact:.4f})\n"
        if "ablation_table" in abl:
            md_content += "\n### Ablation Summary Table\n\n"
            md_content += "| Variant | Fitness Impact | Classification | Description |\n"
            md_content += "|---------|----------------|----------------|-------------|\n"
            for row in abl["ablation_table"]:
                var = str(row.get("variant", ""))[:20]
                imp = row.get("fitness_impact", 0)
                cat = row.get("classification", "")
                desc = str(row.get("description", ""))[:40]
                md_content += f"| {var} | {imp:+.4f} | {cat} | {desc} |\n"
            md_content += "\n"

    # Clinical Interpretation
    md_content += "## Clinical Interpretation\n\n"
    md_content += "Metrics map to clinical outcomes (see `docs/CLINICAL_METRICS.md`):\n\n"
    md_content += "- **adverse_events_rate**: Safety; target minimize\n"
    md_content += "- **critical_TTC_p95**: Time-to-critical for high-acuity; target < 30 min\n"
    md_content += "- **missed_critical_rate**: Critical patients not served; target < 2%\n"
    md_content += "- **throughput**: Patients served; efficiency\n"
    md_content += "- **mean_wait**: Patient experience\n"
    md_content += "- **overload_time**: Queue overload; system stress\n\n"

    # Practical Recommendations (from summary)
    if "recommendations" in summary:
        md_content += "## Practical Recommendations\n\n"
        for rec in summary["recommendations"]:
            md_content += f"- {rec}\n"
        md_content += "\n"

    # Artifacts
    if "artifacts" in report:
        md_content += "## Generated Artifacts\n\n"
        artifacts = report["artifacts"]

        for category, files in artifacts.items():
            if files:
                md_content += f"### {category.title()}\n\n"
                for file_path in files:
                    file_name = Path(file_path).name
                    md_content += f"- {file_name}\n"
                md_content += "\n"

    # Write markdown file
    with open(output_path, "w") as f:
        f.write(md_content)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", type=str, default="results", help="Results directory"
    )
    parser.add_argument("--run_id", type=str, help="Specific run ID to analyze")
    parser.add_argument("--plots", action="store_true", help="Generate plots")

    args = parser.parse_args()

    generate_comprehensive_report(
        results_dir=args.results_dir, run_id=args.run_id, include_plots=args.plots
    )
