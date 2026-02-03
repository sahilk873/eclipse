#!/usr/bin/env python3
"""
Test LLM Mutation Functionality
Tests if LLM mutation is working with your OpenAI API key.
"""

import sys
import uuid
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from env_config import get_openai_api_key


def test_llm_mutation():
    """Test LLM mutation with sample mechanisms."""

    # Check for API key (from .env)
    api_key = get_openai_api_key()
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        print("Add OPENAI_API_KEY to .env in the project root")
        return False

    print("✅ API key found")

    # Import required modules
    try:
        from evolution.llm_mutation import llm_mutate
        from mechanisms.schema import validate_mechanism

        print("✅ Modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Create test mechanisms (global for import testing)
    global test_mechanisms
    test_mechanisms = [
        {
            "info_policy": {"info_mode": "none"},
            "service_policy": {"service_rule": "fifo"},
            "redirect_exit_policy": {
                "redirect_low_risk": False,
                "redirect_mode": "none",
                "reneging_enabled": True,
            },
            "meta": {
                "id": str(uuid.uuid4()),  # Generate valid UUID
                "parent_ids": [],
                "generation": 0,
                "seed_tag": "test",
            },
        },
        {
            "info_policy": {"info_mode": "exact"},
            "service_policy": {"service_rule": "severity_priority"},
            "redirect_exit_policy": {
                "redirect_low_risk": True,
                "redirect_mode": "risk_cutoff",
                "params": {"risk_threshold": 0.3},
                "reneging_enabled": True,
            },
            "meta": {
                "id": str(uuid.uuid4()),  # Generate valid UUID
                "parent_ids": [],
                "generation": 0,
                "seed_tag": "test",
            },
        },
    ]

    # Validate test mechanisms
    for i, mech in enumerate(test_mechanisms):
        valid, errors = validate_mechanism(mech)
        if not valid:
            print(f"❌ Test mechanism {i + 1} invalid: {errors}")
            return False
        print(f"✅ Test mechanism {i + 1} valid")

    # Create test metrics
    test_metrics = [
        {
            "fitness": 15.2,
            "throughput": 12.5,
            "critical_TTC_p95": 18.0,
            "adverse_events_rate": 0.05,
            "overload_time": 45.0,
        },
        {
            "fitness": 8.7,
            "throughput": 9.2,
            "critical_TTC_p95": 32.5,
            "adverse_events_rate": 0.12,
            "overload_time": 78.0,
        },
    ]

    # Create failure analysis
    test_failures = [
        "Mechanism 1: High wait times causing patient dissatisfaction",
        "Mechanism 2: Too many low-risk patients being redirected, overwhelming system capacity",
        "Both mechanisms: Adverse event rates above acceptable threshold",
    ]

    print("\n🧪 Testing LLM mutation...")
    print(f"   Input mechanisms: {len(test_mechanisms)}")
    print(f"   Model to use: gpt-4o-mini (recommended)")

    try:
        # Call LLM mutation
        results = llm_mutate(
            top_mechanisms=test_mechanisms,
            top_metrics_list=test_metrics,
            failure_bullets=test_failures,
            api_key=api_key,
            model="gpt-4o-mini",  # Use smaller, faster model for testing
        )

        if results:
            print(f"✅ LLM mutation SUCCESS! Generated {len(results)} new mechanisms:")

            for i, mechanism in enumerate(results):
                # Validate generated mechanisms
                valid, errors = validate_mechanism(mechanism)
                status = "✅ VALID" if valid else f"❌ INVALID: {errors}"

                print(f"\n   Mechanism {i + 1}: {status}")
                print(
                    f"   Info policy: {mechanism.get('info_policy', {}).get('info_mode', 'unknown')}"
                )
                print(
                    f"   Service rule: {mechanism.get('service_policy', {}).get('service_rule', 'unknown')}"
                )
                print(
                    f"   Redirect mode: {mechanism.get('redirect_exit_policy', {}).get('redirect_mode', 'unknown')}"
                )
                print(f"   ID: {mechanism.get('meta', {}).get('id', 'unknown')}")

            return True
        else:
            print("❌ LLM mutation returned no results")
            return False

    except Exception as e:
        print(f"❌ LLM mutation FAILED: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check if your API key has sufficient credits")
        print("   2. Verify network connectivity")
        print("   3. Try with model='gpt-3.5-turbo' instead")
        print("   4. Check OpenAI API status at status.openai.com")
        return False


def main():
    """Main test function."""
    print("🧪 ECLIPSE LLM Mutation Test")
    print("=" * 50)

    success = test_llm_mutation()

    print("\n" + "=" * 50)
    if success:
        print("🎉 LLM mutation test PASSED!")
        print("✅ Your LLM integration is working correctly")
        print("✅ Ready to use with: python3 run_complete_pipeline.py --use-llm")
    else:
        print("❌ LLM mutation test FAILED!")
        print("🔧 Please fix the issues above before using LLM mutation")

    print("=" * 50)


if __name__ == "__main__":
    main()
