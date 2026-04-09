"""Tests T2.4: Spawning cardinality distribution fitting."""

import numpy as np
import pytest
from scipy import stats as sp_stats

from simoc.ingestion import OCELData
from simoc.discovery.data_structures import (
    BirthDeathTable,
    SpawningProfile,
    TypeClassification,
)
from simoc.discovery.cardinality import compute_spawning_profiles
from simoc.discovery.interaction_graph import classify_types


@pytest.fixture(scope="session")
def spawning_profiles(
    loaded_data: OCELData,
    birth_death: BirthDeathTable,
    type_class: TypeClassification,
) -> dict[tuple[str, str], SpawningProfile]:
    return compute_spawning_profiles(loaded_data, birth_death, type_class)


class TestT2_4_Cardinality:
    def test_spawning_pairs_found(self, spawning_profiles):
        assert ("order", "item") in spawning_profiles
        assert ("order", "delivery") in spawning_profiles

    def test_order_item_raw_counts(self, spawning_profiles):
        """order_1 has 2 items, order_2 has 1 item."""
        profile = spawning_profiles[("order", "item")]
        assert sorted(profile.raw_counts) == [1, 2]

    def test_order_delivery_raw_counts(self, spawning_profiles):
        """Both orders map to delivery_1 → counts = [1, 1]."""
        profile = spawning_profiles[("order", "delivery")]
        assert sorted(profile.raw_counts) == [1, 1]

    def test_ks_test_fitted_distribution(self, spawning_profiles):
        """Sample from fitted dist, KS test vs empirical, p > 0.05."""
        for key, profile in spawning_profiles.items():
            samples = profile.fitted.rvs(size=1000, rng=np.random.default_rng(42))
            empirical = profile.raw_counts
            _, p_value = sp_stats.ks_2samp(samples, empirical)
            assert p_value > 0.05, (
                f"KS test failed for {key}: p={p_value:.4f}"
            )

    def test_aic_finite(self, spawning_profiles):
        for key, profile in spawning_profiles.items():
            assert np.isfinite(profile.fitted.aic), f"AIC not finite for {key}"

    def test_empirical_pmf_sums_to_one(self, spawning_profiles):
        for key, profile in spawning_profiles.items():
            total = sum(profile.fitted.empirical_pmf.values())
            assert total == pytest.approx(1.0), (
                f"PMF sums to {total} for {key}"
            )

    def test_fitted_name_valid(self, spawning_profiles):
        valid_names = {"poisson", "geometric", "nbinom", "empirical"}
        for key, profile in spawning_profiles.items():
            assert profile.fitted.name in valid_names, (
                f"Invalid dist name '{profile.fitted.name}' for {key}"
            )

    def test_rvs_returns_correct_size(self, spawning_profiles):
        for profile in spawning_profiles.values():
            samples = profile.fitted.rvs(size=50)
            assert len(samples) == 50
