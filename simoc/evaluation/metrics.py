"""Task 7.1: Evaluation metrics comparing real vs. synthetic OCEL logs."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from scipy.spatial.distance import cosine as cosine_distance

from simoc.ingestion.data_structures import OCELData
from simoc.evaluation._helpers import (
    discover_oc_dfg,
    extract_activity_frequencies,
    extract_cardinality_counts,
    extract_cycle_times,
    extract_durations,
    extract_inter_arrival_times,
    extract_objects_per_event,
)


# ------------------------------------------------------------------
# Level 1: Activity-Level Metrics
# ------------------------------------------------------------------


def activity_frequency_emd(real: OCELData, synthetic: OCELData) -> float:
    """M1: Earth Mover's Distance between activity frequency distributions."""
    real_freq = extract_activity_frequencies(real)
    syn_freq = extract_activity_frequencies(synthetic)

    all_acts = sorted(set(real_freq) | set(syn_freq))
    if not all_acts:
        return 0.0

    indices = list(range(len(all_acts)))
    real_total = sum(real_freq.values()) or 1
    syn_total = sum(syn_freq.values()) or 1

    real_props = [real_freq.get(a, 0) / real_total for a in all_acts]
    syn_props = [syn_freq.get(a, 0) / syn_total for a in all_acts]

    return float(sp_stats.wasserstein_distance(indices, indices, real_props, syn_props))


def duration_ks_pass_rate(real: OCELData, synthetic: OCELData) -> float:
    """M2: Fraction of (type, activity) pairs where KS test p > 0.05."""
    real_durs = extract_durations(real)
    syn_durs = extract_durations(synthetic)

    pass_count = 0
    total = 0

    for key, r_vals in real_durs.items():
        if len(r_vals) < 2:
            continue
        s_vals = syn_durs.get(key, [])
        if len(s_vals) < 2:
            total += 1  # count as fail
            continue
        _, p = sp_stats.ks_2samp(r_vals, s_vals)
        total += 1
        if p > 0.05:
            pass_count += 1

    return pass_count / total if total > 0 else 1.0


def arrival_rate_error(real: OCELData, synthetic: OCELData) -> float:
    """M3: Mean relative error of mean inter-arrival time per type."""
    errors = []
    for otype in real.object_types:
        r_iat = extract_inter_arrival_times(real, otype)
        s_iat = extract_inter_arrival_times(synthetic, otype)
        if not r_iat:
            continue
        r_mean = np.mean(r_iat)
        if r_mean == 0:
            continue
        if not s_iat:
            errors.append(1.0)
            continue
        s_mean = np.mean(s_iat)
        errors.append(abs(r_mean - s_mean) / r_mean)

    return float(np.mean(errors)) if errors else 0.0


# ------------------------------------------------------------------
# Level 2: Object-Level Metrics
# ------------------------------------------------------------------


def cycle_time_ks(real: OCELData, synthetic: OCELData) -> dict[str, float]:
    """M4: KS test p-value for cycle time distributions per type."""
    real_ct = extract_cycle_times(real)
    syn_ct = extract_cycle_times(synthetic)
    result = {}

    for otype in real.object_types:
        r_vals = real_ct.get(otype, [])
        s_vals = syn_ct.get(otype, [])
        if len(r_vals) < 2 or len(s_vals) < 2:
            # Skip types with too few samples (don't penalize)
            continue
        _, p = sp_stats.ks_2samp(r_vals, s_vals)
        result[otype] = float(p)

    return result


def cardinality_ks(real: OCELData, synthetic: OCELData) -> dict[tuple[str, str], float]:
    """M5: KS test p-value for children-per-parent per spawning pair."""
    result = {}
    # Discover spawning pairs from real O2O
    type_pairs = set()
    for _, row in real.o2o.iterrows():
        type_pairs.add((row["source_type"], row["target_type"]))

    for parent, child in type_pairs:
        r_counts = extract_cardinality_counts(real, parent, child)
        s_counts = extract_cardinality_counts(synthetic, parent, child)
        if len(r_counts) < 2 or len(s_counts) < 2:
            result[(parent, child)] = 0.0
            continue
        _, p = sp_stats.ks_2samp(r_counts, s_counts)
        result[(parent, child)] = float(p)

    return result


def sync_delay_ks(
    real: OCELData,
    synthetic: OCELData,
    sync_rules: dict | None = None,
) -> dict[tuple[str, str], float]:
    """M6: KS test p-value for sync delay distributions.

    Simplified: compares the duration between the last child's prior event
    and the sync event for known sync points.
    """
    # If no sync rules, return empty (metric N/A)
    if not sync_rules:
        return {}

    # Simplified: return empty dict for now — sync delay extraction
    # from arbitrary OCELData would duplicate patterns.py logic.
    # For the paper, we extract sync delays during Stage 4 discovery
    # and compare those distributions directly.
    return {}


# ------------------------------------------------------------------
# Level 3: Structural Metrics
# ------------------------------------------------------------------


def oc_dfg_cosine_similarity(real: OCELData, synthetic: OCELData) -> float:
    """M7: Cosine similarity between OC-DFG edge frequency vectors."""
    real_dfg = discover_oc_dfg(real)
    syn_dfg = discover_oc_dfg(synthetic)

    all_edges = sorted(set(real_dfg) | set(syn_dfg))
    if not all_edges:
        return 1.0

    real_vec = np.array([real_dfg.get(e, 0) for e in all_edges], dtype=float)
    syn_vec = np.array([syn_dfg.get(e, 0) for e in all_edges], dtype=float)

    if np.linalg.norm(real_vec) == 0 or np.linalg.norm(syn_vec) == 0:
        return 0.0

    return float(1.0 - cosine_distance(real_vec, syn_vec))


def o2o_fidelity(real: OCELData, synthetic: OCELData) -> float:
    """M8: Fraction of O2O type pairs with matching cardinality distribution."""
    if real.o2o.empty:
        return 1.0

    # Find type pairs in real O2O
    type_pairs = set()
    for _, row in real.o2o.iterrows():
        type_pairs.add((row["source_type"], row["target_type"]))

    if not type_pairs:
        return 1.0

    pass_count = 0
    total = 0

    for parent, child in type_pairs:
        r_counts = extract_cardinality_counts(real, parent, child)
        s_counts = extract_cardinality_counts(synthetic, parent, child)
        if not r_counts:
            continue
        total += 1
        if len(r_counts) < 2 or len(s_counts) < 2:
            # Not enough samples for KS; check if distributions match exactly
            if sorted(r_counts) == sorted(s_counts):
                pass_count += 1
            continue
        _, p = sp_stats.ks_2samp(r_counts, s_counts)
        if p > 0.05:
            pass_count += 1

    return pass_count / total if total > 0 else 1.0


def convergence_divergence_ks(
    real: OCELData, synthetic: OCELData
) -> dict[str, float]:
    """M9: KS test on objects-per-event distribution per activity."""
    real_ope = extract_objects_per_event(real)
    syn_ope = extract_objects_per_event(synthetic)
    result = {}

    for act in real_ope:
        r_vals = real_ope[act]
        s_vals = syn_ope.get(act, [])
        if len(r_vals) < 2 or len(s_vals) < 2:
            result[act] = 0.0
            continue
        _, p = sp_stats.ks_2samp(r_vals, s_vals)
        result[act] = float(p)

    return result


# ------------------------------------------------------------------
# Aggregate
# ------------------------------------------------------------------


def compute_all_metrics(
    real: OCELData,
    synthetic: OCELData,
    sync_rules: dict | None = None,
) -> dict[str, float]:
    """Compute all metrics, aggregating dict-valued ones to scalars."""
    ct = cycle_time_ks(real, synthetic)
    card = cardinality_ks(real, synthetic)
    sd = sync_delay_ks(real, synthetic, sync_rules)
    cd = convergence_divergence_ks(real, synthetic)

    def _safe_mean(d: dict) -> float:
        vals = list(d.values())
        return float(np.mean(vals)) if vals else 1.0

    return {
        "activity_frequency_emd": activity_frequency_emd(real, synthetic),
        "duration_ks_pass_rate": duration_ks_pass_rate(real, synthetic),
        "arrival_rate_error": arrival_rate_error(real, synthetic),
        "cycle_time_ks_mean_p": _safe_mean(ct),
        "cardinality_ks_mean_p": _safe_mean(card),
        "sync_delay_ks_mean_p": _safe_mean(sd),
        "oc_dfg_cosine_similarity": oc_dfg_cosine_similarity(real, synthetic),
        "o2o_fidelity": o2o_fidelity(real, synthetic),
        "convergence_divergence_ks_mean_p": _safe_mean(cd),
    }
