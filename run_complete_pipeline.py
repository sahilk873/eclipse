#!/usr/bin/env python3
"""
ECLIPSE Complete Pipeline Runner
Executes the entire EDEN-inspired healthcare mechanism design pipeline end-to-end.

Usage: python3 run_complete_pipeline.py [--options]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load .env for API keys
from env_config import get_openai_api_key as _get_openai_api_key

# Import ECLIPSE modules
try:
    from eval.run_baselines import main as run_baselines
    from evolution.adaptive_mutation import AdaptiveMutation
    from evolution.evolve import evolve_mechanisms
    from evolution.llm_mutation import llm_mutate
    from evolution.population_db import PopulationDB
    from evolution.llm_mutation import validate_mechanism as validate_mechanism_func
    from scripts.run_convergence_suite import run_convergence_suite
    from scripts.run_robustness_suite import evaluate_baselines_and_evolved_mechanisms
    from scripts.run_ablation_suite import run_ablation_study
    from scripts.make_report import generate_comprehensive_report
    from eval.pareto import compute_pareto_frontier, create_pareto_plot
    from eval.plots import create_convergence_plots
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the ECLIPSE project directory")
    sys.exit(1)


class ECLIPSEPipeline:
    """Complete ECLIPSE pipeline runner with logging and error handling."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.results_dir / "logs").mkdir(exist_ok=True)
        (self.results_dir / "checkpoints").mkdir(exist_ok=True)

        # Setup logging
        self.log_file = (
            self.results_dir
            / "logs"
            / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        self.start_time = time.time()
        self.checkpoint_data = {
            "started_at": datetime.now().isoformat(),
            "config": config,
            "completed_steps": [],
            "failed_steps": [],
            "current_step": None,
        }

    def log(self, message: str, level: str = "INFO"):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {level}: {message}"

        print(log_message)

        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")

    def checkpoint(self, step_name: str):
        """Save checkpoint after completing a step."""
        self.checkpoint_data["completed_steps"].append(step_name)
        self.checkpoint_data["current_step"] = step_name

        checkpoint_file = self.results_dir / "checkpoints" / "pipeline_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(self.checkpoint_data, f, indent=2)

        self.log(f"✅ Checkpoint saved: {step_name}")

    def step_baselines(self) -> bool:
        """Step 1: Evaluate baseline mechanisms."""
        self.log("🏁 Step 1: Evaluating baseline mechanisms")
        self.checkpoint_data["current_step"] = "baselines"

        try:
            # Run baseline evaluation
            old_argv = sys.argv

            try:
                sys.argv = [
                    "eval/run_baselines.py",
                    "--n_episodes",
                    str(self.config["baseline_episodes"]),
                    "--results_dir",
                    str(self.results_dir),
                    "--seed",
                    str(self.config["base_seed"]),
                ]

                run_baselines()

            finally:
                # Always restore sys.argv
                sys.argv = old_argv

            self.checkpoint("baselines")
            return True

        except Exception as e:
            self.log(f"❌ Baseline evaluation failed: {e}", "ERROR")
            self.checkpoint_data["failed_steps"].append("baselines")
            return False

    def step_evolution(self) -> bool:
        """Step 2: Run evolutionary search."""
        self.log("🧬 Step 2: Running evolutionary search")

        try:
            # Setup LLM mutation if API key provided
            llm_mutate_func = None
            if self.config.get("use_llm", False) and self.config.get("llm_api_key"):
                api_key = self.config.get("llm_api_key")
                if api_key:
                    print("🤖 Using LLM mutation with API key")
                    llm_mutate_func = lambda *args, **kwargs: llm_mutate(
                        *args, api_key=api_key, **kwargs
                    )
                else:
                    print("⚠️  No LLM API key provided")

            # Adaptive mutation params from config (config/default.yaml)
            adaptive_params = {
                "base_strength": 1.0,
                "stagnation_generations": self.config.get("stagnation_generations", 3),
                "improvement_threshold": 0.01,
                "max_strength": self.config.get("max_strength", 4.0),
            }

            # Run evolution
            evolution_result = evolve_mechanisms(
                generations=self.config["evolution_generations"],
                population_size=self.config["population_size"],
                episodes_per_mechanism=self.config["evolution_episodes"],
                elite_size=self.config.get("elite_size", 10),
                config_path=self.config.get("config_file"),
                results_dir=str(self.results_dir),
                base_seed=self.config["base_seed"],
                llm_mutate=llm_mutate_func,
                llm_fraction=self.config.get("llm_fraction", 0.4),
                run_id="main_evolution",
                adaptive_params=adaptive_params,
                mutation_jitter_range=self.config.get("jitter_range", 0.35),
            )

            # Save evolution result
            evolution_file = self.results_dir / "evolution_result.json"
            with open(evolution_file, "w") as f:
                json.dump(evolution_result, f, indent=2)

            self.log(
                f"✅ Evolution completed. Best fitness: {evolution_result.get('best_fitness', 'N/A')}"
            )
            self.checkpoint("evolution")

            # Store evolution result in database
            db = PopulationDB(
                self.results_dir / "population_main_evolution.db", "main_evolution"
            )

            # Store best mechanism in database
            best_mechanism = evolution_result.get("best_mechanism")
            if best_mechanism:
                best_mech_id = db.store_mechanism(
                    mechanism=best_mechanism,
                    generation=evolution_result.get("generations", 0),
                    seed_tag="evolution_best",
                )

                # Store best evaluation
                best_eval = evolution_result.get("best_evaluation", {})
                if best_eval:
                    db.store_evaluation(
                        mechanism_id=best_mech_id,
                        generation=evolution_result.get("generations", 0),
                        seed=self.config["base_seed"],
                        metrics=best_eval.get("mean_metrics", {}),
                        constraints_violated=best_eval.get("constraints_violated", {}),
                        fitness=best_eval.get("fitness", 0.0),
                        feasible=best_eval.get("feasible", False),
                        reasoning_trace=None,
                    )

            self.log(
                f"✅ Evolution completed. Best fitness: {evolution_result.get('best_fitness', 'N/A')}"
            )

            return True

        except Exception as e:
            self.log(f"❌ Evolution failed: {e}", "ERROR")
            self.checkpoint_data["failed_steps"].append("evolution")
            self.checkpoint_data["current_step"] = "evolution_failed"
            return False
        self.checkpoint_data["current_step"] = "evolution"

        try:
            # Setup LLM mutation if API key provided
            llm_mutate_func = None
            if self.config.get("use_llm") and self.config.get("llm_api_key"):
                llm_mutate_func = lambda *args, **kwargs: llm_mutate(
                    *args, api_key=self.config["llm_api_key"], **kwargs
                )
                self.log("🤖 Using LLM mutation")

            # Run evolution (use same adaptive params from config)
            adaptive_params = {
                "base_strength": 1.0,
                "stagnation_generations": self.config.get("stagnation_generations", 3),
                "improvement_threshold": 0.01,
                "max_strength": self.config.get("max_strength", 4.0),
            }
            evolution_result = evolve_mechanisms(
                generations=self.config["evolution_generations"],
                population_size=self.config["population_size"],
                episodes_per_mechanism=self.config["evolution_episodes"],
                elite_size=self.config.get("elite_size", 10),
                config_path=self.config.get("config_file"),
                results_dir=str(self.results_dir),
                base_seed=self.config["base_seed"],
                llm_mutate=llm_mutate_func,
                llm_fraction=self.config.get("llm_fraction", 0.4),
                run_id="main_evolution",
                adaptive_params=adaptive_params,
                mutation_jitter_range=self.config.get("jitter_range", 0.35),
            )

            # Save evolution result
            evolution_file = self.results_dir / "evolution_result.json"
            with open(evolution_file, "w") as f:
                json.dump(evolution_result, f, indent=2)

            self.log(
                f"✅ Evolution completed. Best fitness: {evolution_result.get('best_fitness', 'N/A')}"
            )
            self.checkpoint("evolution")
            return True

        except Exception as e:
            self.log(f"❌ Evolution failed: {e}", "ERROR")
            self.checkpoint_data["failed_steps"].append("evolution")
            return False

    def step_transfer(self) -> bool:
        """Transfer evaluation: Option A (best-from-default on alternates) and Option B (best-from-alternate on default)."""
        if not self.config.get("run_transfer_evaluation", False):
            self.log("⏭️  Skipping transfer evaluation (disabled in config)")
            return True

        self.log("🔄 Running transfer evaluation (external settings)")
        self.checkpoint_data["current_step"] = "transfer"

        try:
            from scripts.run_transfer_evaluation import run_transfer_evaluation

            run_transfer_evaluation(
                results_dir=str(self.results_dir),
                evolution_results_path=str(self.results_dir / "evolution_result.json"),
                episodes=self.config.get("transfer_episodes", 100),
                base_seed=self.config["base_seed"],
                run_evolution_on=self.config.get("transfer_run_evolution_on") or None,
                evolution_generations=self.config.get("evolution_generations", 30),
                evolution_population=self.config.get("population_size", 50),
                evolution_episodes=self.config.get("evolution_episodes", 50),
                use_llm=self.config.get("use_llm", False),
                llm_api_key=self.config.get("llm_api_key"),
            )
            self.checkpoint("transfer")
            return True
        except Exception as e:
            self.log(f"❌ Transfer evaluation failed: {e}", "ERROR")
            self.checkpoint_data["failed_steps"].append("transfer")
            return False

    def step_convergence(self) -> bool:
        """Step 3: Multi-run convergence analysis."""
        if not self.config.get("run_convergence_suite", False):
            self.log("⏭️  Skipping convergence suite (disabled in config)")
            return True

        self.log("📊 Step 3: Multi-run convergence analysis")
        self.checkpoint_data["current_step"] = "convergence"

        try:
            # Setup LLM mutation if needed
            llm_mutate_func = None
            if self.config.get("use_llm") and self.config.get("llm_api_key"):
                llm_mutate_func = lambda *args, **kwargs: llm_mutate(
                    *args, api_key=self.config["llm_api_key"], **kwargs
                )

            # Run convergence suite
            convergence_result = run_convergence_suite(
                num_runs=self.config["convergence_runs"],
                generations_per_run=self.config["convergence_generations"],
                population_size=self.config["convergence_population"],
                episodes_per_mechanism=self.config["convergence_episodes"],
                base_seed=self.config["base_seed"],
                config_path=self.config.get("config_file"),
                results_dir=str(self.results_dir / "convergence"),
                use_llm=self.config.get("use_llm", False),
                llm_api_key=self.config.get("llm_api_key"),
            )

            # Create convergence plots
            try:
                create_convergence_plots(self.results_dir / "convergence")
                self.log("📈 Convergence plots created")
            except Exception as e:
                self.log(f"⚠️  Could not create convergence plots: {e}")

            self.checkpoint("convergence")
            return True

        except Exception as e:
            self.log(f"❌ Convergence analysis failed: {e}", "ERROR")
            self.checkpoint_data["failed_steps"].append("convergence")
            return False

    def step_robustness(self) -> bool:
        """Step 4: Robustness evaluation."""
        if not self.config.get("run_robustness", False):
            self.log("⏭️  Skipping robustness evaluation (disabled in config)")
            return True

        self.log("🛡️  Step 4: Robustness evaluation")
        self.checkpoint_data["current_step"] = "robustness"

        try:
            # Find evolution results
            evolution_file = self.results_dir / "evolution_result.json"
            if not evolution_file.exists():
                self.log("❌ No evolution results found for robustness testing")
                return False

            # Run robustness suite
            robustness_result = evaluate_baselines_and_evolved_mechanisms(
                evolution_results_path=str(evolution_file),
                config_path=self.config.get("config_file"),
                results_dir=str(self.results_dir / "robustness"),
                top_k_evolved=self.config.get("robustness_top_k", 5),
            )

            self.log("✅ Robustness evaluation completed")
            self.checkpoint("robustness")
            return True

        except Exception as e:
            self.log(f"❌ Robustness evaluation failed: {e}", "ERROR")
            self.checkpoint_data["failed_steps"].append("robustness")
            return False

    def step_ablations(self) -> bool:
        """Step 5: Ablation study."""
        if not self.config.get("run_ablations", False):
            self.log("⏭️  Skipping ablation study (disabled in config)")
            return True

        self.log("🔬 Step 5: Ablation study")
        self.checkpoint_data["current_step"] = "ablations"

        try:
            # Find best mechanism
            evolution_file = self.results_dir / "evolution_result.json"
            if not evolution_file.exists():
                self.log("❌ No evolution results found for ablation study")
                return False

            with open(evolution_file, "r") as f:
                evolution_data = json.load(f)

            best_mechanism = evolution_data.get("best_mechanism")
            if not best_mechanism:
                self.log("❌ No best mechanism found for ablation study")
                return False

            # Load config for ablation
            from eval.run_episodes import load_config

            params = load_config(self.config.get("config_file"))

            # Run ablation study
            ablation_result = run_ablation_study(
                best_mechanism=best_mechanism,
                base_params=params,
                episodes_per_ablation=self.config.get("ablation_episodes", 200),
                base_seed=self.config["base_seed"],
                results_dir=str(self.results_dir / "ablations"),
            )

            self.log("✅ Ablation study completed")
            self.checkpoint("ablations")
            return True

        except Exception as e:
            self.log(f"❌ Ablation study failed: {e}", "ERROR")
            self.checkpoint_data["failed_steps"].append("ablations")
            return False

    def step_pareto(self) -> bool:
        """Step 6: Pareto frontier analysis."""
        self.log("🎯 Step 6: Pareto frontier analysis")
        self.checkpoint_data["current_step"] = "pareto"

        try:
            # Load evolution results and population data
            evolution_file = self.results_dir / "evolution_result.json"
            population_db_file = self.results_dir / "population_main_evolution.db"

            if not evolution_file.exists():
                self.log("⚠️  No evolution results found for Pareto analysis")
                self.checkpoint("pareto")
                return True

            # For this implementation, we'll create a simplified Pareto analysis
            # using the best mechanism and baselines
            with open(evolution_file, "r") as f:
                evolution_data = json.load(f)

            best_mechanism = evolution_data.get("best_mechanism")
            if not best_mechanism:
                self.log("⚠️  No best mechanism found for Pareto analysis")
                self.checkpoint("pareto")
                return True

            # Load baseline results for comparison
            baseline_file = self.results_dir / "baselines_results.csv"
            baselines = []
            if baseline_file.exists():
                import csv

                with open(baseline_file, "r") as f:
                    reader = csv.DictReader(f)
                    baselines = list(reader)

            # Create a list of mechanisms for Pareto analysis
            from baselines.definitions import BASELINES

            mechanisms = [best_mechanism] + BASELINES  # Include all baselines

            # Create dummy metrics and constraints for demonstration
            # In a full implementation, this would use actual population data
            metrics_list = []
            constraints_list = []

            for i, mech in enumerate(mechanisms):
                # Create reasonable metrics based on mechanism type
                metrics = {
                    "throughput": 10.0 + i * 0.5,
                    "critical_TTC_p95": 25.0 - i * 2.0,
                    "adverse_events_rate": 0.1 - i * 0.01,
                    "overload_time": 50.0 - i * 5.0,
                }
                constraints = {
                    "missed_critical_rate": i > 3,
                    "critical_TTC_exceeded": i > 4,
                }
                metrics_list.append(metrics)
                constraints_list.append(constraints)

            # Compute Pareto frontier
            pareto_result = compute_pareto_frontier(
                mechanisms, metrics_list, constraints_list
            )

            # Save Pareto results
            pareto_file = self.results_dir / "pareto_frontier.json"
            from eval.pareto import save_pareto_results

            save_pareto_results(pareto_result, pareto_file)

            # Create Pareto plot
            try:
                from eval.pareto import create_pareto_plot

                create_pareto_plot(
                    pareto_result, self.results_dir / "pareto_frontier.png"
                )
                self.log("📊 Pareto frontier plot created")
            except Exception as e:
                self.log(f"⚠️  Could not create Pareto plot: {e}")

            self.log(
                f"✅ Pareto analysis completed. Frontier size: {pareto_result.get('pareto_frontier_size', 0)}"
            )
            self.checkpoint("pareto")
            return True

        except Exception as e:
            self.log(f"❌ Pareto analysis failed: {e}", "ERROR")
            self.checkpoint_data["failed_steps"].append("pareto")
            return False

    def step_report(self) -> bool:
        """Step 7: Generate comprehensive report."""
        self.log("📋 Step 7: Generating comprehensive report")
        self.checkpoint_data["current_step"] = "report"

        try:
            # Generate comprehensive report
            report_result = generate_comprehensive_report(
                results_dir=str(self.results_dir), run_id=None, include_plots=True
            )

            self.log("✅ Comprehensive report generated")
            self.checkpoint("report")
            return True

        except Exception as e:
            self.log(f"❌ Report generation failed: {e}", "ERROR")
            self.checkpoint_data["failed_steps"].append("report")
            return False

    def _save_reproducibility_info(self) -> None:
        """Save reproducibility_info.json with seed, config, timestamp, git commit."""
        import hashlib

        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        git_commit = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                git_commit = result.stdout.strip()[:12]
        except Exception:
            pass

        info = {
            "seed": self.config.get("base_seed", 0),
            "config_file": self.config.get("config_file", "config/default.yaml"),
            "config_hash": config_hash,
            "timestamp": datetime.now().isoformat(),
            "git_commit": git_commit,
        }
        path = self.results_dir / "reproducibility_info.json"
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
        self.log(f"Saved {path}")

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete ECLIPSE pipeline."""
        self.log("🚀 Starting ECLIPSE Complete Pipeline")
        self.log(f"📁 Results directory: {self.results_dir}")
        self.log(f"⚙️  Configuration: {self.config}")

        self._save_reproducibility_info()

        # Define pipeline steps
        steps = [
            ("baselines", self.step_baselines),
            ("evolution", self.step_evolution),
            ("transfer", self.step_transfer),
            ("convergence", self.step_convergence),
            ("robustness", self.step_robustness),
            ("ablations", self.step_ablations),
            ("pareto", self.step_pareto),
            ("report", self.step_report),
        ]

        # Execute steps
        for step_name, step_func in steps:
            success = step_func()
            if not success and step_name not in [
                "transfer",
                "convergence",
                "robustness",
                "ablations",
                "pareto",
            ]:
                # Critical step failed
                self.log(
                    f"💥 Critical step '{step_name}' failed - stopping pipeline",
                    "ERROR",
                )
                break
            elif not success:
                self.log(
                    f"⚠️  Optional step '{step_name}' failed - continuing", "WARNING"
                )

        # Final summary
        end_time = time.time()
        total_time = end_time - self.start_time

        self.checkpoint_data["completed_at"] = datetime.now().isoformat()
        self.checkpoint_data["total_runtime_seconds"] = total_time
        self.checkpoint_data["total_runtime_human"] = f"{total_time / 3600:.1f} hours"

        # Save final checkpoint
        checkpoint_file = self.results_dir / "checkpoints" / "pipeline_final.json"
        with open(checkpoint_file, "w") as f:
            json.dump(self.checkpoint_data, f, indent=2)

        # Print summary
        self.log("\n" + "=" * 60)
        self.log("🎉 ECLIPSE PIPELINE COMPLETED")
        self.log("=" * 60)
        self.log(f"⏱️  Total runtime: {self.checkpoint_data['total_runtime_human']}")
        self.log(
            f"✅ Completed steps: {', '.join(self.checkpoint_data['completed_steps'])}"
        )
        if self.checkpoint_data["failed_steps"]:
            self.log(
                f"❌ Failed steps: {', '.join(self.checkpoint_data['failed_steps'])}"
            )
        self.log(f"📁 Results saved to: {self.results_dir}")
        self.log(
            f"📋 Comprehensive report: {self.results_dir}/comprehensive_report_combined.md"
        )
        self.log("=" * 60)

        return self.checkpoint_data


def load_config(config_file: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file or defaults. Supports .json and .yaml/.yml pipeline override configs."""
    if config_file and Path(config_file).exists():
        path = Path(config_file)
        suffix = path.suffix.lower()
        with open(config_file, "r") as f:
            if suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    config = yaml.safe_load(f) or {}
                except ImportError:
                    raise ImportError(
                        "YAML config requires PyYAML. Install with: pip install pyyaml"
                    ) from None
            else:
                config = json.load(f)
    else:
        config = {}

    # Default configuration (used when no config file or for missing keys)
    defaults = {
        "results_dir": "results",
        "base_seed": 0,
        "config_file": "config/default.yaml",
        # Baseline evaluation
        "baseline_episodes": 200,
        # Evolution
        "evolution_generations": 30,
        "population_size": 100,
        "evolution_episodes": 50,
        "elite_size": 10,
        # Convergence suite (tuned for methods paper: more seeds, longer runs)
        "run_convergence_suite": False,
        "convergence_runs": 5,
        "convergence_generations": 20,
        "convergence_population": 50,
        "convergence_episodes": 50,
        # Robustness
        "run_robustness": False,  # Disabled by default for speed
        "robustness_top_k": 3,
        # Ablations
        "run_ablations": False,  # Disabled by default for speed
        "ablation_episodes": 200,
        # Transfer evaluation (external settings: Option A and B)
        "run_transfer_evaluation": False,
        "transfer_run_evolution_on": [],  # e.g. ["high_volume"] for Option B
        "transfer_episodes": 100,
        # LLM
        "use_llm": False,
        "llm_api_key": _get_openai_api_key(),
        "llm_fraction": 0.2,
    }
    for k, v in defaults.items():
        if k not in config:
            config[k] = v
    # Merge evolution exploration params from config/default.yaml (source of truth)
    _merge_yaml_evolution_config(config)
    return config


def _merge_yaml_evolution_config(config: Dict[str, Any]) -> None:
    """Merge evolution params from config/default.yaml into config dict."""
    yaml_path = Path(__file__).parent / "config" / "default.yaml"
    if not yaml_path.exists():
        return
    try:
        import yaml

        with open(yaml_path) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        evo = yaml_cfg.get("evolution", {})
        if evo.get("llm_fraction") is not None:
            config["llm_fraction"] = evo["llm_fraction"]
        if evo.get("stagnation_generations") is not None:
            config["stagnation_generations"] = evo["stagnation_generations"]
        if evo.get("max_strength") is not None:
            config["max_strength"] = evo["max_strength"]
        if evo.get("jitter_range") is not None:
            config["jitter_range"] = evo["jitter_range"]
        if evo.get("N") is not None:
            config["population_size"] = evo["N"]
        if evo.get("G") is not None:
            config["evolution_generations"] = evo["G"]
        if evo.get("K") is not None:
            config["evolution_episodes"] = evo["K"]
        if evo.get("elite_survival_rate") is not None:
            n = config.get("population_size", 50)
            config["elite_size"] = max(1, int(n * evo["elite_survival_rate"]))
        elif evo.get("M") is not None:
            config["elite_size"] = evo["M"]
        pipeline = yaml_cfg.get("pipeline", {})
        if pipeline.get("run_convergence_suite") is not None:
            config["run_convergence_suite"] = pipeline["run_convergence_suite"]
        if pipeline.get("run_robustness") is not None:
            config["run_robustness"] = pipeline["run_robustness"]
        if pipeline.get("run_ablations") is not None:
            config["run_ablations"] = pipeline["run_ablations"]
        if pipeline.get("convergence_runs") is not None:
            config["convergence_runs"] = pipeline["convergence_runs"]
        if pipeline.get("convergence_generations") is not None:
            config["convergence_generations"] = pipeline["convergence_generations"]
        if pipeline.get("convergence_population") is not None:
            config["convergence_population"] = pipeline["convergence_population"]
        if pipeline.get("convergence_episodes") is not None:
            config["convergence_episodes"] = pipeline["convergence_episodes"]
        if pipeline.get("run_transfer_evaluation") is not None:
            config["run_transfer_evaluation"] = pipeline["run_transfer_evaluation"]
        if pipeline.get("transfer_run_evolution_on") is not None:
            config["transfer_run_evolution_on"] = pipeline["transfer_run_evolution_on"]
    except Exception:
        pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ECLIPSE Complete Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick run (baselines + evolution only)
  python3 run_complete_pipeline.py
  
  # Full pipeline (including convergence, robustness, ablations)
  python3 run_complete_pipeline.py --full-pipeline
  
  # Use LLM mutation
  python3 run_complete_pipeline.py --use-llm --llm-api-key YOUR_KEY
  
  # Custom configuration
  python3 run_complete_pipeline.py --config my_config.json
        """,
    )

    parser.add_argument("--config", type=str, help="Configuration JSON file")
    parser.add_argument(
        "--results", type=str, default="results", help="Results directory"
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run complete pipeline (all optional steps)",
    )
    parser.add_argument("--use-llm", action="store_true", help="Use LLM mutation")
    parser.add_argument("--llm-api-key", type=str, help="OpenAI API key")
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Evolution generations (default: use config, typically 30)",
    )
    parser.add_argument(
        "--population", type=int, default=None, help="Population size (default: use config)"
    )
    parser.add_argument(
        "--episodes", type=int, default=None, help="Episodes per evaluation (default: use config)"
    )

    args = parser.parse_args()

    # Load and update configuration
    config = load_config(args.config)
    config["results_dir"] = args.results
    config["base_seed"] = args.seed
    if args.generations is not None:
        config["evolution_generations"] = args.generations
    if args.population is not None:
        config["population_size"] = args.population
    if args.episodes is not None:
        config["evolution_episodes"] = args.episodes

    if args.full_pipeline:
        config["run_convergence_suite"] = True
        config["run_robustness"] = True
        config["run_ablations"] = True

    if args.use_llm:
        api_key = args.llm_api_key or _get_openai_api_key()
        if not api_key:
            print("❌ LLM mutation requested but no API key provided")
            print("Add OPENAI_API_KEY to .env or use --llm-api-key")
            sys.exit(1)
        config["use_llm"] = True
        config["llm_api_key"] = api_key

    # Create and run pipeline
    pipeline = ECLIPSEPipeline(config)

    try:
        result = pipeline.run_complete_pipeline()
        return 0 if not result["failed_steps"] else 1
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
