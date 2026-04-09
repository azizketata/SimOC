"""Tasks 3.1–3.4: Per-object-type behavioral discovery.

Discovers arrival rates, activity durations, resource pools, and branching
probabilities for each object type.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from simoc.ingestion.data_structures import OCELData
from simoc.discovery.data_structures import (
    ArrivalModel,
    BehavioralProfile,
    BranchingModel,
    BirthDeathTable,
    ContinuousFittedDistribution,
    DurationModel,
    ResourcePool,
    TypeClassification,
    TypeDFG,
)

logger = logging.getLogger(__name__)

# Thresholds for small-sample handling
MIN_SAMPLES_FOR_FIT = 2
MIN_SAMPLES_FOR_STATIONARITY = 20
MIN_SAMPLES_FOR_OUTLIER_REMOVAL = 10

# Candidate continuous distributions: (name, scipy_class, n_params, fit_kwargs)
_CONTINUOUS_CANDIDATES = [
    ("exponential", stats.expon, 1, {"floc": 0}),
    ("gamma", stats.gamma, 2, {"floc": 0}),
    ("lognormal", stats.lognorm, 2, {"floc": 0}),
    ("weibull", stats.weibull_min, 2, {"floc": 0}),
    ("normal", stats.norm, 2, {}),
]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def discover_behavioral(
    data: OCELData,
    birth_death: BirthDeathTable,
    type_classification: TypeClassification,
) -> BehavioralProfile:
    """Orchestrator: discover all behavioral parameters (Tasks 3.1–3.4)."""
    arrival_models = discover_arrival_models(data, birth_death, type_classification)
    duration_models = discover_duration_models(data)
    resource_pools = discover_resource_pools(data)
    branching_models, type_dfgs = discover_branching_models(data)

    logger.info(
        "Behavioral discovery complete: %d arrival models, %d duration models, "
        "%d resource pools, %d branching models.",
        len(arrival_models),
        len(duration_models),
        len(resource_pools),
        len(branching_models),
    )
    return BehavioralProfile(
        arrival_models=arrival_models,
        duration_models=duration_models,
        resource_pools=resource_pools,
        branching_models=branching_models,
        type_dfgs=type_dfgs,
    )


def discover_arrival_models(
    data: OCELData,
    birth_death: BirthDeathTable,
    type_classification: TypeClassification,
) -> dict[str, ArrivalModel]:
    """Task 3.1: Discover arrival rate distributions for root types."""
    models: dict[str, ArrivalModel] = {}

    for otype, role in type_classification.classification.items():
        if role != "root":
            continue

        # Get sorted birth timestamps for this type
        bd_rows = birth_death.df[birth_death.df["object_type"] == otype]
        timestamps = sorted(bd_rows["birth_timestamp"].tolist())
        n_arrivals = len(timestamps)

        if n_arrivals < 2:
            # Not enough arrivals to compute inter-arrival times
            logger.info(
                "Root type '%s' has %d arrivals — insufficient for fitting.",
                otype,
                n_arrivals,
            )
            models[otype] = ArrivalModel(
                object_type=otype,
                is_stationary=True,
                distribution=_make_empirical_continuous(np.array([0.0])),
                piecewise_rates=None,
                n_arrivals=n_arrivals,
                birth_timestamps=timestamps,
            )
            continue

        # Compute inter-arrival gaps in seconds
        gaps = np.array(
            [
                (timestamps[i + 1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]
        )

        # Test stationarity
        is_stationary = _test_stationarity(gaps)

        if is_stationary:
            distribution = _fit_continuous(gaps)
            piecewise = None
        else:
            distribution = _fit_continuous(gaps)  # still provide a global fit
            piecewise = _fit_piecewise_rates(timestamps)

        models[otype] = ArrivalModel(
            object_type=otype,
            is_stationary=is_stationary,
            distribution=distribution,
            piecewise_rates=piecewise,
            n_arrivals=n_arrivals,
            birth_timestamps=timestamps,
        )
        logger.info(
            "Root type '%s': %d arrivals, stationary=%s, best fit=%s",
            otype,
            n_arrivals,
            is_stationary,
            distribution.name,
        )

    return models


def discover_duration_models(
    data: OCELData,
) -> dict[tuple[str, str], DurationModel]:
    """Task 3.2: Discover activity duration distributions per (type, activity)."""
    # Group objects by type
    oid_to_type = data.objects.set_index("object_id")["object_type"].to_dict()

    # Collect durations: for consecutive lifecycle events, duration = ts_{i+1} - ts_i
    durations_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)

    for oid, lifecycle in data.lifecycles.items():
        otype = oid_to_type.get(oid, "")
        for i in range(len(lifecycle) - 1):
            _, activity, ts_curr = lifecycle[i]
            _, _, ts_next = lifecycle[i + 1]
            duration_sec = (ts_next - ts_curr).total_seconds()
            durations_by_key[(otype, activity)].append(duration_sec)

    # Fit distributions
    models: dict[tuple[str, str], DurationModel] = {}
    for (otype, activity), raw_durations in durations_by_key.items():
        samples = np.array(raw_durations)

        # Outlier removal (only if enough samples)
        if len(samples) >= MIN_SAMPLES_FOR_OUTLIER_REMOVAL:
            mean, std = samples.mean(), samples.std()
            if std > 0:
                mask = np.abs(samples - mean) <= 3 * std
                removed = (~mask).sum()
                if removed > 0:
                    logger.info(
                        "Removed %d outliers from (%s, %s).", removed, otype, activity
                    )
                samples = samples[mask]

        distribution = _fit_continuous(samples)
        models[(otype, activity)] = DurationModel(
            object_type=otype,
            activity=activity,
            distribution=distribution,
            n_observations=len(samples),
            raw_durations_seconds=samples.tolist(),
        )

    logger.info("Discovered %d duration models.", len(models))
    return models


def discover_resource_pools(data: OCELData) -> dict[str, ResourcePool]:
    """Task 3.3: Discover resource pools from event attributes."""
    # Check for resource-like columns
    resource_col = None
    for col in data.events.columns:
        if col.lower() in ("resource", "org:resource", "performer", "org:role"):
            resource_col = col
            break

    if resource_col is None:
        logger.info("No resource attribute found. Using infinite-capacity model.")
        return {}

    pools: dict[str, ResourcePool] = {}
    for activity, group in data.events.groupby("activity"):
        resources = sorted(group[resource_col].dropna().unique().tolist())
        if resources:
            pools[activity] = ResourcePool(activity=activity, resources=resources)

    logger.info("Discovered %d resource pools.", len(pools))
    return pools


def discover_branching_models(
    data: OCELData,
) -> tuple[dict[str, BranchingModel], dict[str, TypeDFG]]:
    """Task 3.4: Discover branching probabilities and per-type DFGs."""
    oid_to_type = data.objects.set_index("object_id")["object_type"].to_dict()

    # Collect transitions per type
    transitions: dict[str, dict[tuple[str, str], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    activities_by_type: dict[str, set[str]] = defaultdict(set)

    for oid, lifecycle in data.lifecycles.items():
        otype = oid_to_type.get(oid, "")
        for i in range(len(lifecycle)):
            _, activity, _ = lifecycle[i]
            activities_by_type[otype].add(activity)
            if i < len(lifecycle) - 1:
                _, next_activity, _ = lifecycle[i + 1]
                transitions[otype][(activity, next_activity)] += 1

    branching_models: dict[str, BranchingModel] = {}
    type_dfgs: dict[str, TypeDFG] = {}

    for otype in sorted(activities_by_type.keys()):
        trans = dict(transitions[otype])
        acts = activities_by_type[otype]

        # Compute probabilities: normalize outgoing edges per source activity
        outgoing: dict[str, dict[str, int]] = defaultdict(dict)
        for (src, tgt), count in trans.items():
            outgoing[src][tgt] = count

        probabilities: dict[str, dict[str, float]] = {}
        for src, targets in outgoing.items():
            total = sum(targets.values())
            probabilities[src] = {
                tgt: cnt / total for tgt, cnt in targets.items()
            }

        branching_models[otype] = BranchingModel(
            object_type=otype,
            transition_counts=trans,
            probabilities=probabilities,
        )
        type_dfgs[otype] = TypeDFG(
            object_type=otype,
            activities=acts,
            edges=trans,
            probabilities=probabilities,
        )

    logger.info("Discovered %d branching models.", len(branching_models))
    return branching_models, type_dfgs


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _fit_continuous(samples: np.ndarray) -> ContinuousFittedDistribution:
    """Fit candidate continuous distributions and select best by AIC."""
    n = len(samples)

    # Too few samples or zero variance → empirical
    if n < MIN_SAMPLES_FOR_FIT or np.std(samples) == 0:
        return _make_empirical_continuous(samples)

    # Ensure positive values for constrained distributions
    positive = np.all(samples > 0)

    candidates: list[ContinuousFittedDistribution] = []

    for name, dist_class, k, fit_kwargs in _CONTINUOUS_CANDIDATES:
        # Skip non-negative distributions if data has non-positive values
        if not positive and "floc" in fit_kwargs:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist_class.fit(samples, **fit_kwargs)
                frozen = dist_class(*params)
                ll = frozen.logpdf(samples).sum()

            if not np.isfinite(ll):
                continue

            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll

            candidates.append(
                ContinuousFittedDistribution(
                    name=name,
                    params=_params_to_dict(name, params),
                    aic=aic,
                    bic=bic,
                    n_samples=n,
                    _scipy_dist=frozen,
                    _empirical_samples=samples.copy(),
                )
            )
        except Exception:
            continue

    if not candidates:
        return _make_empirical_continuous(samples)

    return min(candidates, key=lambda c: c.aic)


def _make_empirical_continuous(samples: np.ndarray) -> ContinuousFittedDistribution:
    """Create an empirical continuous distribution (resample fallback)."""
    return ContinuousFittedDistribution(
        name="empirical",
        params={},
        aic=float("inf"),
        bic=float("inf"),
        n_samples=len(samples),
        _scipy_dist=None,
        _empirical_samples=samples.copy(),
    )


def _params_to_dict(name: str, params: tuple) -> dict[str, float]:
    """Convert scipy fit params tuple to a readable dict."""
    if name == "exponential":
        return {"loc": float(params[0]), "scale": float(params[1])}
    elif name == "gamma":
        return {"a": float(params[0]), "loc": float(params[1]), "scale": float(params[2])}
    elif name == "lognormal":
        return {"s": float(params[0]), "loc": float(params[1]), "scale": float(params[2])}
    elif name == "weibull":
        return {"c": float(params[0]), "loc": float(params[1]), "scale": float(params[2])}
    elif name == "normal":
        return {"loc": float(params[0]), "scale": float(params[1])}
    return {f"p{i}": float(p) for i, p in enumerate(params)}


def _test_stationarity(gaps: np.ndarray) -> bool:
    """Test if inter-arrival gaps are stationary via chi-squared test.

    Returns True if stationary (or too few samples to test).
    """
    if len(gaps) < MIN_SAMPLES_FOR_STATIONARITY:
        return True  # Assume stationary with insufficient data

    n_windows = min(4, len(gaps) // 5)
    if n_windows < 2:
        return True

    window_size = len(gaps) // n_windows
    observed = [
        gaps[i * window_size : (i + 1) * window_size].sum()
        for i in range(n_windows)
    ]

    total = sum(observed)
    expected = [total / n_windows] * n_windows

    _, p_value = stats.chisquare(observed, expected)
    return p_value >= 0.05


def _fit_piecewise_rates(
    timestamps: list[pd.Timestamp], block_hours: int = 2
) -> dict[str, float]:
    """Fit piecewise arrival rates by time-of-day blocks."""
    # Extract hour of day for each timestamp
    hours = [ts.hour + ts.minute / 60 for ts in timestamps]

    rates: dict[str, float] = {}
    for start in range(0, 24, block_hours):
        end = start + block_hours
        label = f"{start:02d}:00-{end:02d}:00"
        count = sum(1 for h in hours if start <= h < end)
        block_seconds = block_hours * 3600
        rates[label] = count / block_seconds if block_seconds > 0 else 0.0

    return rates
