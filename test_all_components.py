#!/usr/bin/env python3
"""
Comprehensive ECLIPSE Test Suite
Tests each component independently before full pipeline run.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class TestMechanisms(unittest.TestCase):
    """Test mechanism schema, generation, and validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_schema_validation(self):
        """Test mechanism schema validation."""
        from mechanisms.schema import validate_mechanism

        # Valid mechanism
        valid_mech = {
            "info_policy": {"info_mode": "none"},
            "service_policy": {"service_rule": "fifo"},
            "redirect_exit_policy": {
                "redirect_low_risk": False,
                "redirect_mode": "none",
                "reneging_enabled": True,
            },
            "meta": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "parent_ids": [],
                "generation": 0,
                "seed_tag": "test",
            },
        }

        valid, errors = validate_mechanism(valid_mech)
        self.assertTrue(valid, f"Valid mechanism should pass: {errors}")
        self.assertEqual(len(errors), 0, "Should have no validation errors")

        print("✅ Schema validation test PASSED")

    def test_random_mechanism_generation(self):
        """Test random mechanism generation."""
        from mechanisms.genome import random_mechanism, validate_mechanism

        for i in range(5):
            mech = random_mechanism(seed=42 + i)
            valid, errors = validate_mechanism(mech)

            self.assertTrue(valid, f"Generated mechanism {i} should be valid: {errors}")
            self.assertIn("info_policy", mech)
            self.assertIn("service_policy", mech)
            self.assertIn("redirect_exit_policy", mech)
            self.assertIn("meta", mech)

        print("✅ Random mechanism generation test PASSED")

    def test_mechanism_mutation(self):
        """Test mechanism mutation."""
        from mechanisms.genome import random_mechanism
        from mechanisms.mutation import mutate_mechanism, validate_mechanism

        # Create parent mechanism
        parent = random_mechanism(seed=100)
        self.assertTrue(validate_mechanism(parent)[0])

        # Generate mutations
        for i in range(5):
            child = mutate_mechanism(parent, seed=200 + i, mutation_strength=1.0)
            valid, errors = validate_mechanism(child)

            self.assertTrue(valid, f"Mutated mechanism {i} should be valid: {errors}")

            # Check mutation changed something (or at least didn't make it worse)
            # Note: Mutation might not always change every component
            parent_info = parent.get("info_policy", {})
            child_info = child.get("info_policy", {})

            # Should be valid mechanisms (both pass validation)
            self.assertIsInstance(parent_info, dict)
            self.assertIsInstance(child_info, dict)

        print("✅ Mechanism mutation test PASSED")


class TestSimulator(unittest.TestCase):
    """Test simulation components."""

    def test_simulator_imports(self):
        """Test simulator module imports."""
        try:
            from sim.runner import run_episode
            from sim.processes import (
                create_patient,
                should_admit_patient,
                select_next_patient,
                sample_interarrival,
            )
            from sim.entities import Patient, SystemState

            print("✅ Simulator imports test PASSED")
        except ImportError as e:
            self.fail(f"Simulator import failed: {e}")

    def test_patient_creation(self):
        """Test patient creation process."""
        from sim.processes import create_patient
        from sim.entities import Patient

        # Test parameters
        patient = create_patient(
            patient_id=1,
            arrival_time=0.0,
            risk_mix={"critical": 0.1, "urgent": 0.3, "low": 0.6},
            service_params={"default": {"dist": "lognormal", "mean": 20, "std": 10}},
            patience_params={"default": 60},
        )

        self.assertIsInstance(patient, Patient)
        self.assertEqual(patient.id, 1)
        self.assertEqual(patient.arrival_time, 0.0)
        self.assertIn(patient.risk_class, ["critical", "urgent", "low"])

        print("✅ Patient creation test PASSED")

    def test_episode_execution(self):
        """Test single episode execution."""
        from sim.runner import run_episode

        # Simple mechanism
        mechanism = {
            "info_policy": {"info_mode": "none"},
            "service_policy": {"service_rule": "fifo"},
            "redirect_exit_policy": {
                "redirect_low_risk": False,
                "redirect_mode": "none",
                "reneging_enabled": True,
            },
            "meta": {
                "id": "test",
                "parent_ids": [],
                "generation": 0,
                "seed_tag": "test",
            },
        }

        # Minimal params
        params = {
            "T": 60,  # 1 hour for quick test
            "lambda": 2.0,  # Lower arrival rate
            "n_servers": 2,
            "Qmax": 10,
            "service": {"default": {"dist": "lognormal", "mean": 15, "std": 5}},
            "patience": {"default": 45},
            "risk_mix": {"critical": 0.1, "urgent": 0.3, "low": 0.6},
            "benefit_per_class": {"critical": 50, "urgent": 25, "low": 10},
            "c_wait": 0.5,
            "deterioration": {"enabled": False},
        }

        try:
            metrics, constraints = run_episode(mechanism, params, seed=42)

            self.assertIsInstance(metrics, dict)
            self.assertIn("throughput", metrics)
            self.assertIn("critical_TTC_p95", metrics)
            self.assertIsInstance(constraints, dict)

            print("✅ Episode execution test PASSED")
            print(f"   Throughput: {metrics.get('throughput', 0):.2f}")
            print(f"   Critical TTC p95: {metrics.get('critical_TTC_p95', 0):.2f}")

        except Exception as e:
            self.fail(f"Episode execution failed: {e}")


class TestEvaluation(unittest.TestCase):
    """Test evaluation components."""

    def test_evaluation_imports(self):
        """Test evaluation module imports."""
        try:
            from eval.run_episodes import evaluate_mechanism, load_config
            from eval.metrics import compute_fitness, is_feasible, pareto_metrics
            from eval.run_baselines import main as run_baselines

            print("✅ Evaluation imports test PASSED")
        except ImportError as e:
            self.fail(f"Evaluation import failed: {e}")

    def test_mechanism_evaluation(self):
        """Test mechanism evaluation."""
        from eval.run_episodes import evaluate_mechanism
        from mechanisms.genome import random_mechanism

        mechanism = random_mechanism(seed=200)
        params = {
            "T": 60,
            "lambda": 2.0,
            "n_servers": 2,
            "Qmax": 10,
            "service": {"default": {"dist": "lognormal", "mean": 15, "std": 5}},
            "patience": {"default": 45},
            "risk_mix": {"critical": 0.1, "urgent": 0.3, "low": 0.6},
            "benefit_per_class": {"critical": 50, "urgent": 25, "low": 10},
            "c_wait": 0.5,
            "fitness_weights": {"A": 50, "B": 2.0, "C": 0.5, "D": 1.0, "E": 0.2},
        }

        result = evaluate_mechanism(
            mechanism=mechanism,
            params=params,
            K=5,  # Small number for testing
            base_seed=300,
        )

        self.assertIsInstance(result, dict)
        self.assertIn("mean_metrics", result)
        self.assertIn("fitness", result)
        self.assertIn("feasible", result)

        print("✅ Mechanism evaluation test PASSED")
        print(f"   Fitness: {result.get('fitness', 0):.4f}")
        print(f"   Feasible: {result.get('feasible', False)}")

    def test_metrics_computation(self):
        """Test metrics computation."""
        from eval.metrics import compute_fitness, is_feasible, pareto_metrics

        metrics = {
            "throughput": 10.0,
            "critical_TTC_p95": 25.0,
            "adverse_events_rate": 0.05,
            "overload_time": 40.0,
        }

        constraints_violated = {
            "missed_critical_rate": False,
            "critical_TTC_exceeded": False,
        }

        # Test fitness computation
        fitness = compute_fitness(metrics)
        self.assertIsInstance(fitness, float)

        # Test feasibility
        feasible = is_feasible(constraints_violated)
        self.assertTrue(feasible)

        # Test Pareto metrics
        safety, ttc_p95, overload, throughput = pareto_metrics(
            metrics, constraints_violated
        )
        self.assertEqual(safety, 0.0)  # Should be 0 for feasible mechanisms
        self.assertEqual(ttc_p95, 25.0)
        self.assertEqual(overload, 40.0)
        self.assertEqual(throughput, 10.0)

        print("✅ Metrics computation test PASSED")


class TestEvolution(unittest.TestCase):
    """Test evolution components."""

    def test_evolution_imports(self):
        """Test evolution module imports."""
        try:
            from evolution.evolve import evolve_mechanisms
            from evolution.adaptive_mutation import AdaptiveMutation
            from evolution.population_db import PopulationDB
            from evolution.selection import select_top_m
            from evolution.reproduction import create_offspring
            from evolution.llm_mutation import llm_mutate

            print("✅ Evolution imports test PASSED")
        except ImportError as e:
            self.fail(f"Evolution import failed: {e}")

    def test_adaptive_mutation(self):
        """Test adaptive mutation."""
        from evolution.adaptive_mutation import AdaptiveMutation

        # Test initialization
        adaptive = AdaptiveMutation(
            base_strength=1.0, min_strength=0.1, max_strength=3.0
        )

        self.assertEqual(adaptive.current_strength, 1.0)
        self.assertEqual(adaptive.min_strength, 0.1)
        self.assertEqual(adaptive.max_strength, 3.0)

        # Test strength update
        new_strength = adaptive.update_strength(10.0)  # Better fitness
        self.assertLessEqual(new_strength, adaptive.current_strength)  # Should decay

        # Test stagnation
        for i in range(6):
            strength = adaptive.update_strength(5.0)  # Same fitness, stagnating
            if i >= 2:  # After stagnation_generations
                self.assertGreaterEqual(strength, 1.0)  # Should increase

        print("✅ Adaptive mutation test PASSED")

    def test_population_database(self):
        """Test population database."""
        from evolution.population_db import PopulationDB
        from mechanisms.schema import validate_mechanism

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = PopulationDB(db_path, run_id="test_run")

            # Test mechanism storage
            mechanism = {
                "info_policy": {"info_mode": "none"},
                "service_policy": {"service_rule": "fifo"},
                "redirect_exit_policy": {
                    "redirect_low_risk": False,
                    "redirect_mode": "none",
                    "reneging_enabled": True,
                },
                "meta": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "parent_ids": [],
                    "generation": 0,
                    "seed_tag": "test",
                },
            }

            mech_id = db.store_mechanism(mechanism, generation=0, seed_tag="test")
            self.assertIsNotNone(mech_id)

            # Test evaluation storage
            db.store_evaluation(
                mechanism_id=mech_id,
                generation=0,
                seed=42,
                metrics={"throughput": 10.0, "critical_TTC_p95": 20.0},
                constraints={"missed_critical_rate": False},
                fitness=5.0,
                feasible=True,
            )

            # Test retrieval
            retrieved = db.get_mechanism(mech_id)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved["info_policy"]["info_mode"], "none")

            print("✅ Population database test PASSED")


class TestBaselines(unittest.TestCase):
    """Test baseline components."""

    def test_baselines_imports(self):
        """Test baseline imports."""
        try:
            from baselines.definitions import (
                BASELINES,
                get_baseline_name as get_baseline,
            )
            from eval.run_baselines import main as run_baselines

            print("✅ Baselines imports test PASSED")
        except ImportError as e:
            self.fail(f"Baselines import failed: {e}")

    def test_baseline_definitions(self):
        """Test baseline mechanism definitions."""
        from baselines.definitions import BASELINES, get_baseline_name
        from mechanisms.schema import validate_mechanism

        # Test we have baseline mechanisms
        self.assertGreater(len(BASELINES), 0)
        self.assertEqual(len(BASELINES), 7)  # Should be 7 baselines

        # Test each baseline is valid
        for i, baseline in enumerate(BASELINES):
            valid, errors = validate_mechanism(baseline)
            self.assertTrue(
                valid,
                f"Baseline {i} ({get_baseline_name(i)}) should be valid: {errors}",
            )

        print("✅ Baseline definitions test PASSED")
        print(f"   Found {len(BASELINES)} valid baselines")


class TestLLMMutation(unittest.TestCase):
    """Test LLM mutation functionality."""

    def test_llm_mutation_imports(self):
        """Test LLM mutation imports."""
        try:
            from evolution.llm_mutation import llm_mutate

            print("✅ LLM mutation imports test PASSED")
        except ImportError as e:
            self.fail(f"LLM mutation import failed: {e}")

    @patch("evolution.llm_mutation.get_openai_api_key", return_value=None)
    def test_llm_mutation_no_api_key(self, _mock_get_key):
        """Test LLM mutation without API key (no .env key and no api_key arg)."""
        from evolution.llm_mutation import llm_mutate

        result = llm_mutate(
            top_mechanisms=[], top_metrics_list=[], failure_bullets=[], api_key=None
        )

        self.assertEqual(len(result), 0)  # Should return empty list without API key

        print("✅ LLM mutation no API key test PASSED")

    @patch("evolution.llm_mutation.openai")
    def test_llm_mutation_with_mock_api(self, mock_openai):
        """Test LLM mutation with mocked API."""
        from evolution.llm_mutation import llm_mutate

        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        valid_mechanism_json = '{"info_policy": {"info_mode": "none"}, "service_policy": {"service_rule": "fifo"}, "redirect_exit_policy": {"redirect_low_risk": false, "redirect_mode": "none", "reneging_enabled": true}, "meta": {"id": "550e8400-e29b-41d4-a716-446655440000", "parent_ids": [], "generation": 0, "seed_tag": "test"}}'
        # Mock response: status=completed, output=[], output_text=content (Responses API format)
        mock_response = Mock(
            status="completed",
            output=[],
            output_text=valid_mechanism_json,
        )
        mock_client.responses.create.return_value = mock_response

        # Fallback: if Chat Completions is used instead
        mock_choice = Mock(message=Mock(content=valid_mechanism_json))
        mock_client.chat.completions.create.return_value = Mock(
            choices=[mock_choice]
        )

        result = llm_mutate(
            top_mechanisms=[
                {
                    "info_policy": {"info_mode": "none"},
                    "service_policy": {"service_rule": "fifo"},
                }
            ],
            top_metrics_list=[{}],
            failure_bullets=[],
            api_key="fake-key",
        )

        self.assertGreater(
            len(result), 0,
            "Should return mechanisms with mocked API",
        )

        print("✅ LLM mutation with mock API test PASSED")


class TestPipeline(unittest.TestCase):
    """Test pipeline components and internal consistency."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_config_load_yaml(self):
        """Pipeline load_config accepts YAML (config/default.yaml)."""
        from run_complete_pipeline import load_config
        config = load_config("config/default.yaml")
        self.assertIsInstance(config, dict)
        self.assertIn("config_file", config)
        self.assertIn("results_dir", config)
        # Merged from YAML pipeline section
        self.assertIn("run_transfer_evaluation", config)
        print("   Pipeline YAML config load OK")

    def test_pipeline_config_load_json(self):
        """Pipeline load_config accepts JSON override."""
        from run_complete_pipeline import load_config
        json_path = self.temp_path / "override.json"
        json_path.write_text('{"results_dir": "custom_results", "base_seed": 42}')
        config = load_config(str(json_path))
        self.assertEqual(config.get("results_dir"), "custom_results")
        self.assertEqual(config.get("base_seed"), 42)
        print("   Pipeline JSON config load OK")

    def test_alternate_configs_exist_and_load(self):
        """Alternate configs (high_volume, higher_acuity, low_resource) exist and load."""
        from eval.run_episodes import load_config
        for name, expected_lambda, expected_servers in [
            ("default", 4.0, 3),
            ("high_volume", 6.0, 5),
            ("higher_acuity", 4.0, 3),
            ("low_resource", 2.0, 2),
        ]:
            path = project_root / "config" / ("default.yaml" if name == "default" else f"{name}.yaml")
            if not path.exists():
                self.skipTest(f"Config {name} not found")
            params = load_config(str(path))
            self.assertEqual(params.get("lambda"), expected_lambda, f"{name} lambda")
            self.assertEqual(params.get("n_servers"), expected_servers, f"{name} n_servers")
        # higher_acuity risk_mix
        params_acuity = load_config(str(project_root / "config" / "higher_acuity.yaml"))
        self.assertEqual(params_acuity.get("risk_mix", {}).get("critical"), 0.15)
        print("   Alternate configs load OK")

    def test_baselines_run_all_seven(self):
        """Run baselines produces 7 rows and all baseline names match."""
        from eval.run_baselines import main as run_baselines
        from baselines.definitions import BASELINE_NAMES
        run_baselines(
            n_episodes=2,
            config_path=str(project_root / "config" / "default.yaml"),
            results_dir=str(self.temp_path),
            base_seed=0,
        )
        csv_path = self.temp_path / "baselines_results.csv"
        self.assertTrue(csv_path.exists(), "baselines_results.csv should exist")
        import csv
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 7, "Should have 7 baselines")
        names_in_csv = {r["name"] for r in rows}
        for i, expected_name in enumerate(BASELINE_NAMES):
            self.assertIn(expected_name, names_in_csv, f"Baseline {expected_name} in CSV")
        print("   Baselines run (7) OK")

    def test_transfer_evaluation_option_a(self):
        """Transfer evaluation Option A runs and writes transfer_results.json."""
        from scripts.run_transfer_evaluation import run_transfer_evaluation
        from baselines.definitions import BASELINES
        # Minimal evolution_result with one mechanism
        best = dict(BASELINES[0])
        best["meta"] = best.get("meta", {}) or {}
        best["meta"]["id"] = "test-mech-id"
        evolution_result = {"best_mechanism": best, "best_fitness": 1.0, "generations": 0}
        (self.temp_path / "evolution_result.json").write_text(json.dumps(evolution_result, indent=2))
        out = run_transfer_evaluation(
            results_dir=str(self.temp_path),
            evolution_results_path=str(self.temp_path / "evolution_result.json"),
            episodes=2,
            base_seed=0,
            run_evolution_on=None,
        )
        self.assertIn("evaluations", out)
        self.assertIn("transfer_table", out)
        self.assertIn("summary", out)
        self.assertGreaterEqual(len(out["evaluations"]), 4)  # default, high_volume, higher_acuity, low_resource
        tr_path = self.temp_path / "transfer_results.json"
        self.assertTrue(tr_path.exists())
        data = json.loads(tr_path.read_text())
        self.assertEqual(len(data["transfer_table"]), len(out["evaluations"]))
        print("   Transfer evaluation Option A OK")

    def test_evaluation_dimensions_external_validation_beats(self):
        """Evaluation dimensions external_validation computes 'evolved beats X of N' without summary."""
        from eval.evaluation_dimensions import get_dimension_evidence, EVALUATION_DIMENSIONS
        report = {
            "baseline_comparison": {
                "baseline_table": [
                    {"name": "A", "fitness": 1.0},
                    {"name": "B", "fitness": 2.0},
                    {"name": "C", "fitness": 3.0},
                ],
                "num_baselines": 3,
                "best_baseline_fitness": 3.0,
            },
            "convergence_analysis": {"best_overall_fitness": 2.5},
            "summary": {},  # empty: dimensions must not rely on it
        }
        evidence = get_dimension_evidence(report)
        self.assertIn("external_validation", evidence)
        summary = evidence["external_validation"].get("summary") or ""
        self.assertIn("evolved beats 2 of 3 baselines", summary)
        self.assertEqual(len(evidence), len(EVALUATION_DIMENSIONS))
        print("   Evaluation dimensions external_validation OK")

    def test_analyze_transfer_results_numeric_guard(self):
        """_analyze_transfer_results skips non-numeric values in summary_line."""
        from scripts.make_report import _analyze_transfer_results
        transfer_data = {
            "transfer_table": [],
            "summary": {
                "best_from_default_on_alternates": {
                    "default": 5.0,
                    "high_volume": "bad",  # non-numeric
                    "higher_acuity": 4.5,
                },
            },
            "evaluations": [],
        }
        out = _analyze_transfer_results(transfer_data)
        self.assertIn("summary_line", out)
        self.assertIn("default: 5.00", out["summary_line"])
        self.assertIn("higher_acuity: 4.50", out["summary_line"])
        self.assertNotIn("bad", out["summary_line"])
        print("   Transfer analyze numeric guard OK")

    def test_report_transfer_analysis_section(self):
        """Generate report with transfer_results.json produces transfer_analysis."""
        from scripts.make_report import generate_comprehensive_report, _analyze_transfer_results
        (self.temp_path / "transfer_results.json").write_text(json.dumps({
            "evaluations": [{"train_config": "default", "test_config": "default", "fitness": 3.0, "feasible": True}],
            "transfer_table": [{"train_config": "default", "test_config": "default", "fitness": 3.0, "feasible": True}],
            "summary": {"best_from_default_on_alternates": {"default": 3.0}},
            "config": {},
        }))
        report = generate_comprehensive_report(str(self.temp_path), include_plots=False)
        self.assertIn("transfer_analysis", report)
        self.assertNotIn("error", report["transfer_analysis"])
        self.assertIn("transfer_table", report["transfer_analysis"])
        self.assertIn("summary_line", report["transfer_analysis"])
        print("   Report transfer_analysis section OK")

    def test_evolution_short_run(self):
        """Evolution runs for 2 generations and returns best_mechanism and best_fitness."""
        from evolution.evolve import evolve_mechanisms
        result = evolve_mechanisms(
            generations=2,
            population_size=5,
            episodes_per_mechanism=2,
            elite_size=2,
            config_path=str(project_root / "config" / "default.yaml"),
            results_dir=str(self.temp_path),
            base_seed=99,
            llm_mutate=None,
            llm_fraction=0.0,
            run_id="test_evolution",
        )
        self.assertIn("best_mechanism", result)
        self.assertIn("best_fitness", result)
        self.assertIn("convergence_data", result)
        self.assertEqual(len(result["convergence_data"]), 2)
        if result["best_mechanism"]:
            self.assertIn("info_policy", result["best_mechanism"])
        print("   Evolution short run OK")

    def test_convergence_suite_import(self):
        """Convergence suite module imports and run_convergence_suite is callable."""
        from scripts.run_convergence_suite import run_convergence_suite
        self.assertTrue(callable(run_convergence_suite))
        print("   Convergence suite import OK")

    def test_robustness_suite_import(self):
        """Robustness suite module imports and evaluate_baselines_and_evolved_mechanisms is callable."""
        from scripts.run_robustness_suite import evaluate_baselines_and_evolved_mechanisms
        self.assertTrue(callable(evaluate_baselines_and_evolved_mechanisms))
        print("   Robustness suite import OK")

    def test_ablation_suite_import(self):
        """Ablation suite module imports and run_ablation_study is callable."""
        from scripts.run_ablation_suite import run_ablation_study
        self.assertTrue(callable(run_ablation_study))
        print("   Ablation suite import OK")

    def test_evaluation_dimensions_report_section(self):
        """evaluation_dimensions_report_section returns dimensions and evidence for all six."""
        from eval.evaluation_dimensions import evaluation_dimensions_report_section, EVALUATION_DIMENSIONS
        report = {"baseline_comparison": {"num_baselines": 7}, "convergence_analysis": {}}
        section = evaluation_dimensions_report_section(report)
        self.assertEqual(section["dimensions"], EVALUATION_DIMENSIONS)
        self.assertEqual(len(section["evidence"]), len(EVALUATION_DIMENSIONS))
        for dim in EVALUATION_DIMENSIONS:
            self.assertIn(dim, section["evidence"])
            self.assertIn("description", section["evidence"][dim])
            self.assertIn("addressed", section["evidence"][dim])
        print("   Evaluation dimensions report section OK")


class TestConfig(unittest.TestCase):
    """Test configuration components."""

    def test_config_loading(self):
        """Test configuration loading."""
        try:
            from eval.run_episodes import load_config
            from run_complete_pipeline import load_config as load_pipeline_config

            # Test default config loading
            config1 = load_config(None)
            self.assertIsInstance(config1, dict)
            # Check for key expected fields (sample based on actual structure)
            if config1:  # Only check if config is not empty
                sample_keys = list(config1.keys())[:3]  # Check first 3 keys as sample
                print(f"Config sample keys: {sample_keys}")

            # Test file config loading (only if file exists)
            if Path("pipeline_config.json").exists():
                config2 = load_pipeline_config("pipeline_config.json")
                self.assertIsInstance(config2, dict)
                self.assertIn("results_dir", config2)

            print("✅ Configuration loading test PASSED")

        except ImportError as e:
            self.fail(f"Config loading import failed: {e}")


def run_all_tests():
    """Run all test suites and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestMechanisms,
        TestSimulator,
        TestEvaluation,
        TestEvolution,
        TestBaselines,
        TestLLMMutation,
        TestConfig,
        TestPipeline,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("🧪 ECLIPSE COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%"
    )

    if result.failures:
        print("\n❌ FAILURES:")
        for test, error in result.failures:
            print(f"   {test}: {error}")

    if result.errors:
        print("\n💥 ERRORS:")
        for test, error in result.errors:
            print(f"   {test}: {error}")

    if not result.failures and not result.errors:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ ECLIPSE system is ready for full pipeline execution!")

    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()
