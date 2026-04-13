"""SimOC Streamlit Demo — One page, three sections, one scroll."""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.set_page_config(page_title="SimOC Demo", layout="wide")

# ── Title ──────────────────────────────────────────────────────────
st.title("SimOC")
st.markdown(
    "**Agent-based simulation of object-centric processes, "
    "discovered from OCEL 2.0 event logs.**"
)

# ── Cached discovery pipeline ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading benchmark log and discovering process model (~30 s)...")
def run_discovery():
    from simoc.ingestion import load_ocel, compute_summary
    from simoc.discovery.interaction_graph import discover_interaction_graph
    from simoc.discovery.cardinality import compute_spawning_profiles
    from simoc.discovery.behavioral import discover_behavioral
    from simoc.discovery.patterns import discover_patterns
    from simoc.discovery.data_structures import DiscoveryResult

    data = load_ocel("data/order-management.sqlite")
    summary = compute_summary(data)
    bd, oig, tc = discover_interaction_graph(data)
    sp = compute_spawning_profiles(data, bd, tc)
    bp = discover_behavioral(data, bd, tc)
    ip = discover_patterns(data, bd, tc)
    dr = DiscoveryResult(bd, oig, tc, sp)
    return data, summary, dr, bp, ip, sp

data, summary, dr, bp, ip, sp = run_discovery()
tc = dr.type_classification

# ── Plotting helpers ───────────────────────────────────────────────
TYPE_COLORS = {
    "customers": "#e74c3c", "orders": "#2ecc71", "items": "#3498db",
    "packages": "#f39c12", "products": "#9b59b6", "employees": "#1abc9c",
    "delivery": "#e74c3c", "order": "#2ecc71", "item": "#3498db",
}

def plot_lifecycle_timeline(d, title, max_objects=60):
    oid_to_type = d.objects.set_index("object_id")["object_type"].to_dict()
    t0 = d.events["timestamp"].min()
    type_order = sorted(set(oid_to_type.values()))
    type_y = {t: i + 1 for i, t in enumerate(type_order)}

    fig, ax = plt.subplots(figsize=(12, max(3, len(type_order) * 0.8)))
    count = 0
    for oid, lc in d.lifecycles.items():
        if count >= max_objects:
            break
        otype = oid_to_type.get(oid, "")
        color = TYPE_COLORS.get(otype, "#888")
        y = type_y.get(otype, 0) + (hash(oid) % 100) / 250
        hours = [(ts - t0).total_seconds() / 3600 for _, _, ts in lc]
        ax.plot(hours, [y] * len(hours), color=color, alpha=0.4, linewidth=1)
        ax.scatter(hours, [y] * len(hours), color=color, s=8, zorder=5)
        count += 1

    ax.set_yticks(list(type_y.values()))
    ax.set_yticklabels(list(type_y.keys()), fontsize=10)
    ax.set_xlabel("Time (hours)")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    return fig


def plot_radar(m_simoc, m_flat, m_indep):
    labels = ["Activity\nEMD", "OC-DFG\nsimilarity", "O2O\nfidelity",
              "Cardinality\nKS", "Conv./div.\nKS"]
    keys = ["activity_frequency_emd", "oc_dfg_cosine_similarity",
            "o2o_fidelity", "cardinality_ks_mean_p",
            "convergence_divergence_ks_mean_p"]

    def to_radar(m):
        vals = []
        for k in keys:
            v = m[k]
            if k in ("activity_frequency_emd",):
                vals.append(max(0, 1.0 - v))
            else:
                vals.append(v)
        return vals

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    for vals, name, color, ls in [
        (to_radar(m_simoc), "SimOC", "#e74c3c", "-"),
        (to_radar(m_flat), "Flat (single-type)", "#95a5a6", "--"),
        (to_radar(m_indep), "Independent", "#3498db", "-."),
    ]:
        v = vals + vals[:1]
        ax.plot(angles, v, color=color, linewidth=2, linestyle=ls, label=name)
        ax.fill(angles, v, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.50", "0.75"], fontsize=7, color="gray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title("Simulation Fidelity (higher = better)", fontsize=11,
                 fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — THE INPUT
# ══════════════════════════════════════════════════════════════════
st.header("1. The Input — Real OCEL 2.0 Event Log")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Events", f"{summary.num_events:,}")
c2.metric("Objects", f"{summary.num_objects:,}")
c3.metric("Object Types", summary.num_object_types)
c4.metric("O2O Relations", f"{summary.num_o2o_relations:,}")

col_left, col_right = st.columns([2, 1])
with col_left:
    fig_timeline = plot_lifecycle_timeline(data, "Object Lifecycles Over Time")
    st.pyplot(fig_timeline)
    plt.close(fig_timeline)
with col_right:
    st.markdown("**Objects per Type**")
    type_df = summary.per_object_type[["object_type", "count"]].sort_values(
        "count", ascending=False
    )
    st.bar_chart(type_df.set_index("object_type")["count"])

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — WHAT SIMOC DISCOVERS
# ══════════════════════════════════════════════════════════════════
st.header("2. What SimOC Discovers")

# Type classification table
rows = []
for t in sorted(tc.classification):
    rows.append({
        "Type": t,
        "Role": tc.classification[t].upper(),
        "Parent": tc.parent_map.get(t, "—"),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Interaction patterns in expanders
pat_col1, pat_col2 = st.columns(2)

with pat_col1:
    with st.expander("Spawning Pairs", expanded=True):
        for (pt, ct), profile in sorted(sp.items()):
            st.markdown(
                f"**{pt}** → **{ct}**: "
                f"mean {np.mean(profile.raw_counts):.1f} per parent "
                f"({profile.fitted.name})"
            )
        if not sp:
            st.caption("No spawning pairs discovered.")

    with st.expander("Synchronization Rules"):
        for k, r in sorted(ip.synchronization_rules.items()):
            st.markdown(
                f"**{r.activity}** waits for **{r.synced_type}** "
                f"(condition: {r.condition}, "
                f"mean delay: {np.mean(r.raw_sync_delays)/3600:.1f}h, "
                f"n={r.n_instances})"
            )
        if not ip.synchronization_rules:
            st.caption("No synchronization rules discovered.")

with pat_col2:
    with st.expander("Binding Policies", expanded=True):
        for k, r in sorted(ip.binding_policies.items()):
            st.markdown(
                f"**{r.source_type}** ← **{r.target_type}** "
                f"at *{r.binding_activity}* (n={r.n_instances})"
            )
        if not ip.binding_policies:
            st.caption("No binding policies discovered.")

    with st.expander("Release Rules"):
        for k, r in sorted(ip.release_rules.items()):
            st.markdown(
                f"**({r.type_1}, {r.type_2})** release at "
                f"*{r.release_activity}* ({r.release_condition})"
            )
        if not ip.release_rules:
            st.caption("No release rules discovered.")

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — SIMULATE & COMPARE
# ══════════════════════════════════════════════════════════════════
st.header("3. Simulate & Compare")

if st.button("Run Simulation", type="primary"):
    with st.spinner("Simulating (SimOC + 2 baselines)..."):
        from simoc.simulation.runner import SimulationRunner
        from simoc.evaluation.baselines import (
            build_flat_simod_runner, build_independent_runner,
        )
        from simoc.evaluation._helpers import sim_result_to_oceldata
        from simoc.evaluation.metrics import compute_all_metrics

        duration = (
            data.events["timestamp"].max() - data.events["timestamp"].min()
        ).total_seconds()
        duration = min(duration, 3600 * 24 * 30)

        runner = SimulationRunner(dr, bp, ip, real_data=data)
        result = runner.run(duration=duration, seed=42)
        syn = sim_result_to_oceldata(result)

        flat = build_flat_simod_runner(data, dr, bp, case_type="orders")
        indep = build_independent_runner(data, dr, bp)
        flat_result = flat.run(duration=duration, seed=42)
        indep_result = indep.run(duration=duration, seed=42)

        m_simoc = compute_all_metrics(data, syn)
        m_flat = compute_all_metrics(data, sim_result_to_oceldata(flat_result))
        m_indep = compute_all_metrics(data, sim_result_to_oceldata(indep_result))

    # Side-by-side stats
    st.subheader("Real vs. Synthetic")
    s1, s2 = st.columns(2)
    s1.metric("Real Events", f"{len(data.events):,}")
    s2.metric("Synthetic Events", f"{len(syn.events):,}")
    s1.metric("Real Objects", f"{len(data.objects):,}")
    s2.metric("Synthetic Objects", f"{len(syn.objects):,}")
    s1.metric("Real O2O", f"{len(data.o2o):,}")
    s2.metric("Synthetic O2O", f"{len(syn.o2o):,}")

    # Radar chart
    st.subheader("Fidelity: SimOC vs. Baselines")
    fig_radar = plot_radar(m_simoc, m_flat, m_indep)
    st.pyplot(fig_radar)
    plt.close(fig_radar)

    st.success(
        "SimOC is the **only method** that produces O2O relational structure "
        "(fidelity, cardinality, convergence/divergence > 0)."
    )

    # Metrics table
    st.subheader("Detailed Metrics")
    metric_names = [
        "activity_frequency_emd", "oc_dfg_cosine_similarity",
        "o2o_fidelity", "cardinality_ks_mean_p",
        "convergence_divergence_ks_mean_p",
    ]
    table_rows = []
    for k in metric_names:
        table_rows.append({
            "Metric": k.replace("_", " ").title(),
            "SimOC": f"{m_simoc[k]:.3f}",
            "Flat": f"{m_flat[k]:.3f}",
            "Independent": f"{m_indep[k]:.3f}",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True,
                 hide_index=True)
else:
    st.info("Click **Run Simulation** to generate a synthetic log and compare against baselines.")
