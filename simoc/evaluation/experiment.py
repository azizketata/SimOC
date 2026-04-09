"""Task 7.3: Experiment runner for multi-seed evaluation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from simoc.ingestion.data_structures import OCELData
from simoc.simulation.runner import SimulationRunner
from simoc.evaluation._helpers import sim_result_to_oceldata
from simoc.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an evaluation experiment."""

    duration: float = 3600 * 24 * 30  # 30 days
    n_runs: int = 50
    seeds: list[int] | None = None
    start_timestamp: str | None = None


@dataclass
class MethodResult:
    """Results for one method across all runs."""

    method_name: str
    per_run_metrics: list[dict[str, float]] = field(default_factory=list)
    run_times: list[float] = field(default_factory=list)

    def mean_metrics(self) -> dict[str, float]:
        if not self.per_run_metrics:
            return {}
        keys = self.per_run_metrics[0].keys()
        return {k: float(np.mean([m[k] for m in self.per_run_metrics])) for k in keys}

    def std_metrics(self) -> dict[str, float]:
        if not self.per_run_metrics:
            return {}
        keys = self.per_run_metrics[0].keys()
        return {k: float(np.std([m[k] for m in self.per_run_metrics])) for k in keys}

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.per_run_metrics)


@dataclass
class ExperimentResult:
    """Full experiment results: all methods, all metrics."""

    method_results: dict[str, MethodResult]
    config: ExperimentConfig

    def summary_table(self) -> pd.DataFrame:
        """Rows = methods, columns = 'metric (mean +/- std)'."""
        rows = []
        for name, mr in self.method_results.items():
            means = mr.mean_metrics()
            stds = mr.std_metrics()
            row = {"method": name}
            for k in means:
                row[k] = f"{means[k]:.4f} +/- {stds[k]:.4f}"
            rows.append(row)
        return pd.DataFrame(rows).set_index("method")

    def significance_tests(self, reference: str = "simoc") -> pd.DataFrame:
        """Paired Wilcoxon signed-rank test per metric per baseline vs reference."""
        if reference not in self.method_results:
            return pd.DataFrame()

        ref = self.method_results[reference]
        ref_df = ref.to_dataframe()
        rows = []

        for name, mr in self.method_results.items():
            if name == reference:
                continue
            baseline_df = mr.to_dataframe()
            for metric in ref_df.columns:
                ref_vals = ref_df[metric].values
                base_vals = baseline_df[metric].values

                # Align lengths
                n = min(len(ref_vals), len(base_vals))
                if n < 5:
                    rows.append({
                        "baseline": name,
                        "metric": metric,
                        "test_statistic": np.nan,
                        "p_value": np.nan,
                        "significant": False,
                    })
                    continue

                ref_v = ref_vals[:n]
                base_v = base_vals[:n]
                diffs = ref_v - base_v

                try:
                    if np.all(diffs == 0):
                        stat, p = 0.0, 1.0
                    else:
                        stat, p = sp_stats.wilcoxon(ref_v, base_v)
                except Exception:
                    try:
                        stat, p = sp_stats.ttest_rel(ref_v, base_v)
                    except Exception:
                        stat, p = np.nan, np.nan

                rows.append({
                    "baseline": name,
                    "metric": metric,
                    "test_statistic": float(stat),
                    "p_value": float(p),
                    "significant": p < 0.05 if np.isfinite(p) else False,
                })

        return pd.DataFrame(rows)


def run_experiment(
    real_data: OCELData,
    methods: dict[str, SimulationRunner],
    config: ExperimentConfig | None = None,
    sync_rules: dict | None = None,
) -> ExperimentResult:
    """Run all methods for n_runs seeds and compute metrics."""
    if config is None:
        config = ExperimentConfig()

    seeds = config.seeds or list(range(config.n_runs))
    method_results: dict[str, MethodResult] = {}

    for method_name, runner in methods.items():
        logger.info("Running method '%s' (%d seeds)...", method_name, len(seeds))
        mr = MethodResult(method_name=method_name)

        for seed in seeds:
            t0 = time.perf_counter()
            try:
                sim_result = runner.run(duration=config.duration, seed=seed)
                syn_data = sim_result_to_oceldata(sim_result, config.start_timestamp)
                metrics = compute_all_metrics(real_data, syn_data, sync_rules)
            except Exception as e:
                logger.warning(
                    "Method '%s' seed=%d failed: %s", method_name, seed, e
                )
                # Return NaN metrics for failed runs
                metrics = {k: np.nan for k in [
                    "activity_frequency_emd", "duration_ks_pass_rate",
                    "arrival_rate_error", "cycle_time_ks_mean_p",
                    "cardinality_ks_mean_p", "sync_delay_ks_mean_p",
                    "oc_dfg_cosine_similarity", "o2o_fidelity",
                    "convergence_divergence_ks_mean_p",
                ]}

            elapsed = time.perf_counter() - t0
            mr.per_run_metrics.append(metrics)
            mr.run_times.append(elapsed)

        method_results[method_name] = mr
        logger.info(
            "Method '%s' complete: mean time=%.2fs",
            method_name,
            np.mean(mr.run_times),
        )

    return ExperimentResult(method_results=method_results, config=config)
