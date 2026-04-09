"""Task 7.4: Visualization functions for evaluation results.

Requires optional dependencies: matplotlib, networkx.
Install with: pip install simoc[viz]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from simoc.evaluation.experiment import ExperimentResult

logger = logging.getLogger(__name__)


def plot_comparison_table(
    result: ExperimentResult,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Render and optionally save the comparison results table."""
    table = result.summary_table()
    if output_path:
        table.to_csv(str(output_path))
        logger.info("Saved comparison table to %s", output_path)
    return table


def plot_sync_delay_distributions(
    delays_by_method: dict[str, list[float]],
    output_path: str | Path | None = None,
) -> None:
    """Overlaid histograms of sync delay distributions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, delays in delays_by_method.items():
        if delays:
            ax.hist(delays, bins=30, alpha=0.4, label=name, density=True)
    ax.set_xlabel("Sync delay (seconds)")
    ax.set_ylabel("Density")
    ax.set_title("Synchronization Delay Distribution")
    ax.legend()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Saved sync delay plot to %s", output_path)
    plt.close(fig)


def plot_o2o_fidelity_bar(
    result: ExperimentResult,
    output_path: str | Path | None = None,
) -> None:
    """Bar chart of O2O fidelity per method with error bars."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Skipping plot.")
        return

    methods = []
    means = []
    stds = []
    for name, mr in result.method_results.items():
        m = mr.mean_metrics()
        s = mr.std_metrics()
        if "o2o_fidelity" in m:
            methods.append(name)
            means.append(m["o2o_fidelity"])
            stds.append(s["o2o_fidelity"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(methods, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_ylabel("O2O Fidelity")
    ax.set_title("O2O Relation Fidelity by Method")
    ax.set_ylim(0, 1.1)

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Saved O2O fidelity plot to %s", output_path)
    plt.close(fig)
