"""Task 2.4: Spawning cardinality distribution fitting."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import numpy as np
from scipy import optimize, stats

from simoc.ingestion.data_structures import OCELData
from simoc.discovery.data_structures import (
    BirthDeathTable,
    FittedDistribution,
    SpawningProfile,
    TypeClassification,
)

logger = logging.getLogger(__name__)


def compute_spawning_profiles(
    data: OCELData,
    birth_death: BirthDeathTable,
    type_classification: TypeClassification,
) -> dict[tuple[str, str], SpawningProfile]:
    """For each (parent, child) spawning pair, fit a cardinality distribution."""
    profiles: dict[tuple[str, str], SpawningProfile] = {}

    for child_type, parent_type in type_classification.parent_map.items():
        parent_to_children = _group_children_by_parent(
            data, birth_death, parent_type, child_type
        )

        # Raw cardinality counts: children per parent
        counts = [len(children) for children in parent_to_children.values()]
        if not counts:
            logger.warning(
                "No parent-child instances found for (%s, %s). Skipping.",
                parent_type,
                child_type,
            )
            continue

        fitted = _fit_cardinality(counts)
        attr_deps = _check_attribute_dependency(
            data,
            {pid: len(ch) for pid, ch in parent_to_children.items()},
            parent_type,
        )

        key = (parent_type, child_type)
        profiles[key] = SpawningProfile(
            parent_type=parent_type,
            child_type=child_type,
            raw_counts=counts,
            fitted=fitted,
            attribute_dependencies=attr_deps,
        )
        logger.info(
            "Spawning (%s -> %s): counts=%s, best fit=%s (AIC=%.2f)",
            parent_type,
            child_type,
            counts,
            fitted.name,
            fitted.aic,
        )

    return profiles


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _group_children_by_parent(
    data: OCELData,
    birth_death: BirthDeathTable,
    parent_type: str,
    child_type: str,
) -> dict[str, list[str]]:
    """Group child objects by their parent. Returns {parent_oid -> [child_oids]}.

    Strategy 1: Use O2O relations for direct parent-child links.
    Strategy 2: Fall back to birth co-objects if O2O is empty.
    """
    parent_to_children: dict[str, list[str]] = defaultdict(list)

    # Get all parent object IDs
    parent_oids = set(
        data.objects.loc[
            data.objects["object_type"] == parent_type, "object_id"
        ]
    )

    # Strategy 1: O2O relations
    if len(data.o2o) > 0:
        # Check both directions: parent->child or child->parent in O2O
        fwd = data.o2o[
            (data.o2o["source_type"] == parent_type)
            & (data.o2o["target_type"] == child_type)
        ]
        rev = data.o2o[
            (data.o2o["source_type"] == child_type)
            & (data.o2o["target_type"] == parent_type)
        ]

        found = False
        for _, row in fwd.iterrows():
            parent_to_children[row["source_object_id"]].append(
                row["target_object_id"]
            )
            found = True
        for _, row in rev.iterrows():
            parent_to_children[row["target_object_id"]].append(
                row["source_object_id"]
            )
            found = True

        if found:
            # Ensure parents with 0 children are included
            for pid in parent_oids:
                if pid not in parent_to_children:
                    parent_to_children[pid] = []
            return dict(parent_to_children)

    # Strategy 2: Birth co-objects
    child_rows = birth_death.df[birth_death.df["object_type"] == child_type]
    for _, row in child_rows.iterrows():
        for co_oid, co_otype in row["birth_co_objects"]:
            if co_otype == parent_type:
                parent_to_children[co_oid].append(row["object_id"])

    # Ensure parents with 0 children are included
    for pid in parent_oids:
        if pid not in parent_to_children:
            parent_to_children[pid] = []

    return dict(parent_to_children)


def _fit_cardinality(counts: list[int]) -> FittedDistribution:
    """Fit candidate discrete distributions and select best by AIC."""
    counts_arr = np.array(counts)
    n = len(counts_arr)

    # Empirical PMF (always computed)
    counter = Counter(counts)
    empirical_pmf = {k: v / n for k, v in sorted(counter.items())}

    # If too few samples, return empirical only
    if n < 2:
        return FittedDistribution(
            name="empirical",
            params={},
            aic=float("inf"),
            bic=float("inf"),
            empirical_pmf=empirical_pmf,
            n_samples=n,
            _scipy_dist=None,
        )

    candidates: list[FittedDistribution] = []

    # --- Poisson ---
    try:
        mu = float(counts_arr.mean())
        if mu > 0:
            dist = stats.poisson(mu)
            ll = dist.logpmf(counts_arr).sum()
            if np.isfinite(ll):
                candidates.append(
                    FittedDistribution(
                        name="poisson",
                        params={"mu": mu},
                        aic=2 * 1 - 2 * ll,
                        bic=1 * np.log(n) - 2 * ll,
                        empirical_pmf=empirical_pmf,
                        n_samples=n,
                        _scipy_dist=dist,
                    )
                )
    except Exception:
        pass

    # --- Geometric (shifted to support {0, 1, 2, ...}) ---
    try:
        mean_val = float(counts_arr.mean())
        if mean_val >= 0:
            # geom(p) has support {1,2,...}, so use loc=-1 to shift to {0,1,...}
            # But if min(counts) >= 1, use standard support
            if counts_arr.min() >= 1:
                p = 1.0 / (1.0 + mean_val - 1.0) if mean_val > 1 else 0.999
                p = max(1e-10, min(p, 1.0 - 1e-10))
                dist = stats.geom(p)
            else:
                p = 1.0 / (1.0 + mean_val) if mean_val > 0 else 0.999
                p = max(1e-10, min(p, 1.0 - 1e-10))
                dist = stats.geom(p, loc=-1)
            ll = dist.logpmf(counts_arr).sum()
            if np.isfinite(ll):
                candidates.append(
                    FittedDistribution(
                        name="geometric",
                        params={"p": p},
                        aic=2 * 1 - 2 * ll,
                        bic=1 * np.log(n) - 2 * ll,
                        empirical_pmf=empirical_pmf,
                        n_samples=n,
                        _scipy_dist=dist,
                    )
                )
    except Exception:
        pass

    # --- Negative binomial (only if n >= 3 and variance > 0) ---
    if n >= 3 and counts_arr.var() > 0:
        try:
            mean_val = float(counts_arr.mean())
            var_val = float(counts_arr.var(ddof=1))

            if var_val > mean_val and mean_val > 0:
                # Method of moments initial guess
                p_init = mean_val / var_val
                n_init = mean_val * p_init / (1 - p_init)
                n_init = max(0.1, n_init)
                p_init = max(1e-10, min(p_init, 1.0 - 1e-10))
            else:
                n_init = max(1.0, mean_val)
                p_init = 0.5

            def neg_ll(params):
                nn, pp = params
                if nn <= 0 or pp <= 0 or pp >= 1:
                    return 1e10
                return -stats.nbinom.logpmf(counts_arr, nn, pp).sum()

            result = optimize.minimize(
                neg_ll,
                x0=[n_init, p_init],
                method="Nelder-Mead",
                options={"maxiter": 1000},
            )

            if result.success or result.fun < 1e9:
                nb_n, nb_p = result.x
                if nb_n > 0 and 0 < nb_p < 1:
                    dist = stats.nbinom(nb_n, nb_p)
                    ll = -result.fun
                    if np.isfinite(ll):
                        candidates.append(
                            FittedDistribution(
                                name="nbinom",
                                params={"n": float(nb_n), "p": float(nb_p)},
                                aic=2 * 2 - 2 * ll,
                                bic=2 * np.log(n) - 2 * ll,
                                empirical_pmf=empirical_pmf,
                                n_samples=n,
                                _scipy_dist=dist,
                            )
                        )
        except Exception:
            pass

    # --- Empirical fallback ---
    k_emp = len(empirical_pmf) - 1  # degrees of freedom
    k_emp = max(k_emp, 1)
    # Log-likelihood of empirical: sum(log(pmf[x]) for x in counts)
    emp_ll = sum(np.log(empirical_pmf[x]) for x in counts)
    candidates.append(
        FittedDistribution(
            name="empirical",
            params={},
            aic=2 * k_emp - 2 * emp_ll,
            bic=k_emp * np.log(n) - 2 * emp_ll,
            empirical_pmf=empirical_pmf,
            n_samples=n,
            _scipy_dist=None,
        )
    )

    # Select best by AIC
    best = min(candidates, key=lambda c: c.aic)
    return best


def _check_attribute_dependency(
    data: OCELData,
    parent_child_counts: dict[str, int],
    parent_type: str,
) -> dict[str, float] | None:
    """Check if cardinality correlates with parent attributes (p < 0.05)."""
    parent_df = data.objects[data.objects["object_type"] == parent_type].copy()

    # Identify numeric attribute columns (exclude object_id, object_type)
    attr_cols = [
        c
        for c in parent_df.columns
        if c not in ("object_id", "object_type")
        and parent_df[c].dtype.kind in ("i", "f")
    ]

    if not attr_cols or len(parent_child_counts) < 3:
        return None

    # Align counts with parent DataFrame
    parent_df = parent_df[parent_df["object_id"].isin(parent_child_counts)]
    parent_df["_cardinality"] = parent_df["object_id"].map(parent_child_counts)

    significant: dict[str, float] = {}
    for col in attr_cols:
        valid = parent_df[[col, "_cardinality"]].dropna()
        if len(valid) < 3:
            continue
        corr, pval = stats.spearmanr(valid[col], valid["_cardinality"])
        if pval < 0.05:
            significant[col] = float(corr)

    return significant if significant else None
