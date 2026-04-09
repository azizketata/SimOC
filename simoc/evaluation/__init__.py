"""Evaluation and baseline comparison (Stage 7)."""

from simoc.evaluation.baselines import (
    build_flat_simod_runner,
    build_independent_runner,
    build_random_binding_runner,
)
from simoc.evaluation.experiment import (
    ExperimentConfig,
    ExperimentResult,
    MethodResult,
    run_experiment,
)
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
    sync_delay_ks,
)

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "MethodResult",
    "activity_frequency_emd",
    "arrival_rate_error",
    "build_flat_simod_runner",
    "build_independent_runner",
    "build_random_binding_runner",
    "cardinality_ks",
    "compute_all_metrics",
    "convergence_divergence_ks",
    "cycle_time_ks",
    "duration_ks_pass_rate",
    "o2o_fidelity",
    "oc_dfg_cosine_similarity",
    "run_experiment",
    "sync_delay_ks",
]
