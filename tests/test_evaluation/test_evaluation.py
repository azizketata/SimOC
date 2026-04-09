"""Tests for Stage 7: Evaluation metrics, baselines, and experiment runner."""

import numpy as np
import pytest

from simoc.ingestion.data_structures import OCELData
from simoc.evaluation.metrics import (
    activity_frequency_emd,
    arrival_rate_error,
    cardinality_ks,
    compute_all_metrics,
    convergence_divergence_ks,
    cycle_time_ks,
    duration_ks_pass_rate,
    o2o_fidelity,
    oc_dfg_cosine_similarity,
)
from simoc.evaluation.baselines import (
    build_flat_simod_runner,
    build_independent_runner,
    build_random_binding_runner,
)
from simoc.evaluation.experiment import ExperimentConfig, run_experiment


# ------------------------------------------------------------------
# Metric unit tests: self-comparison and range checks
# ------------------------------------------------------------------

class TestMetricSelfComparison:
    """Comparing a log to itself should give perfect scores."""

    def test_emd_self_zero(self, loaded_data: OCELData):
        assert activity_frequency_emd(loaded_data, loaded_data) == pytest.approx(0.0)

    def test_duration_ks_self_perfect(self, loaded_data: OCELData):
        assert duration_ks_pass_rate(loaded_data, loaded_data) == pytest.approx(1.0)

    def test_arrival_error_self_zero(self, loaded_data: OCELData):
        assert arrival_rate_error(loaded_data, loaded_data) == pytest.approx(0.0)

    def test_cycle_time_ks_self(self, loaded_data: OCELData):
        result = cycle_time_ks(loaded_data, loaded_data)
        # Types with enough samples should have p=1.0
        for otype, p in result.items():
            assert p == pytest.approx(1.0), f"Type {otype} self-comparison p={p}"

    def test_oc_dfg_self_one(self, loaded_data: OCELData):
        sim = oc_dfg_cosine_similarity(loaded_data, loaded_data)
        assert sim == pytest.approx(1.0)

    def test_o2o_fidelity_self_one(self, loaded_data: OCELData):
        assert o2o_fidelity(loaded_data, loaded_data) == pytest.approx(1.0)


class TestMetricRanges:
    """All metrics should return values in expected ranges."""

    def test_emd_nonnegative(self, loaded_data, synthetic_data):
        assert activity_frequency_emd(loaded_data, synthetic_data) >= 0.0

    def test_emd_symmetric(self, loaded_data, synthetic_data):
        fwd = activity_frequency_emd(loaded_data, synthetic_data)
        rev = activity_frequency_emd(synthetic_data, loaded_data)
        assert fwd == pytest.approx(rev, abs=1e-6)

    def test_duration_ks_in_range(self, loaded_data, synthetic_data):
        rate = duration_ks_pass_rate(loaded_data, synthetic_data)
        assert 0.0 <= rate <= 1.0

    def test_arrival_error_nonneg(self, loaded_data, synthetic_data):
        assert arrival_rate_error(loaded_data, synthetic_data) >= 0.0

    def test_oc_dfg_in_range(self, loaded_data, synthetic_data):
        sim = oc_dfg_cosine_similarity(loaded_data, synthetic_data)
        assert 0.0 <= sim <= 1.0

    def test_o2o_fidelity_in_range(self, loaded_data, synthetic_data):
        f = o2o_fidelity(loaded_data, synthetic_data)
        assert 0.0 <= f <= 1.0


class TestComputeAllMetrics:
    def test_returns_all_keys(self, loaded_data, synthetic_data):
        result = compute_all_metrics(loaded_data, synthetic_data)
        expected = {
            "activity_frequency_emd", "duration_ks_pass_rate",
            "arrival_rate_error", "cycle_time_ks_mean_p",
            "cardinality_ks_mean_p", "sync_delay_ks_mean_p",
            "oc_dfg_cosine_similarity", "o2o_fidelity",
            "convergence_divergence_ks_mean_p",
        }
        assert set(result.keys()) == expected

    def test_all_values_finite(self, loaded_data, synthetic_data):
        result = compute_all_metrics(loaded_data, synthetic_data)
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"


# ------------------------------------------------------------------
# Baseline tests
# ------------------------------------------------------------------

class TestBaselines:
    def test_flat_simod_runs(self, loaded_data, discovery_result, behavioral_profile):
        runner = build_flat_simod_runner(
            loaded_data, discovery_result, behavioral_profile, case_type="order"
        )
        result = runner.run(duration=36000, seed=42)
        assert len(result.events) > 0
        types = {o.object_type for o in result.objects}
        assert types == {"order"}

    def test_independent_runs(self, loaded_data, discovery_result, behavioral_profile):
        runner = build_independent_runner(
            loaded_data, discovery_result, behavioral_profile
        )
        result = runner.run(duration=36000, seed=42)
        assert len(result.events) > 0
        assert len(result.o2o_relations) == 0

    def test_random_binding_runs(
        self, discovery_result, behavioral_profile, interaction_patterns
    ):
        runner = build_random_binding_runner(
            discovery_result, behavioral_profile, interaction_patterns
        )
        result = runner.run(duration=36000, seed=42)
        assert len(result.events) > 0

    def test_baselines_no_crash_multiple_seeds(
        self, loaded_data, discovery_result, behavioral_profile, interaction_patterns
    ):
        runners = {
            "flat": build_flat_simod_runner(
                loaded_data, discovery_result, behavioral_profile, "order"
            ),
            "indep": build_independent_runner(
                loaded_data, discovery_result, behavioral_profile
            ),
            "random": build_random_binding_runner(
                discovery_result, behavioral_profile, interaction_patterns
            ),
        }
        for name, runner in runners.items():
            for seed in range(3):
                result = runner.run(duration=36000, seed=seed)
                assert len(result.events) > 0, f"{name} seed={seed} failed"


# ------------------------------------------------------------------
# Experiment and significance tests
# ------------------------------------------------------------------

class TestExperiment:
    def test_small_experiment_runs(
        self, loaded_data, sim_runner, discovery_result,
        behavioral_profile, interaction_patterns,
    ):
        """Run a small 3-seed experiment with all methods."""
        flat = build_flat_simod_runner(
            loaded_data, discovery_result, behavioral_profile, "order"
        )
        indep = build_independent_runner(
            loaded_data, discovery_result, behavioral_profile
        )
        random = build_random_binding_runner(
            discovery_result, behavioral_profile, interaction_patterns
        )

        methods = {
            "simoc": sim_runner,
            "flat_simod": flat,
            "independent": indep,
            "random_binding": random,
        }

        config = ExperimentConfig(duration=36000, n_runs=3, seeds=[0, 1, 2])
        result = run_experiment(loaded_data, methods, config)

        # Check all methods have results
        assert len(result.method_results) == 4
        for name, mr in result.method_results.items():
            assert len(mr.per_run_metrics) == 3, f"{name} has {len(mr.per_run_metrics)} runs"

        # Summary table should work
        table = result.summary_table()
        assert len(table) == 4

        # Significance tests should work
        sig = result.significance_tests("simoc")
        assert len(sig) > 0


# ------------------------------------------------------------------
# T7.6: Reproducibility
# ------------------------------------------------------------------

class TestT7_6_Reproducibility:
    def test_same_seed_same_metrics(self, sim_runner, loaded_data):
        from simoc.evaluation._helpers import sim_result_to_oceldata

        r1 = sim_runner.run(duration=36000, seed=99)
        r2 = sim_runner.run(duration=36000, seed=99)

        d1 = sim_result_to_oceldata(r1)
        d2 = sim_result_to_oceldata(r2)

        m1 = compute_all_metrics(loaded_data, d1)
        m2 = compute_all_metrics(loaded_data, d2)

        for key in m1:
            assert m1[key] == pytest.approx(m2[key]), (
                f"Metric {key} differs: {m1[key]} vs {m2[key]}"
            )
