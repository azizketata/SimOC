# SimOC

Agent-based simulation discovery for object-centric processes from OCEL 2.0 event logs.

SimOC automatically discovers per-object-type agent behaviors and cross-type interaction patterns (spawning, synchronization, binding, batching, release) from real event data, then simulates forward to produce synthetic OCEL 2.0 traces.

## The Gap

Existing simulation discovery approaches either model a *single object type* or use *process-model-based* semantics. No existing approach models *multiple interacting object types as autonomous agents*:

|                              | Single Object Type         | Multiple Interacting Types (Object-Centric) |
|------------------------------|----------------------------|----------------------------------------------|
| **Process-model-based sim.** | Simod (Chapela-Campa et al.) | Knopp et al. (ICPM 2023)                   |
| **Agent-based sim.**         | AgentSimulator (Kirchdorfer et al.) | **SimOC (this project)**            |

- **Simod** discovers BPMN models from single-case XES logs. One object type, no O2O relations.
- **AgentSimulator** models resources as agents operating on a single object type. Handles concurrent cases, but agents are workers — not the objects themselves.
- **Knopp et al.** discovers OCPN-based simulations from OCEL 2.0 using token semantics — not autonomous agents with interaction patterns.
- **SimOC** models object types as agents (orders, items, packages) that spawn, synchronize, bind, batch, and release each other. First agent-based approach for object-centric simulation discovery.

| Feature | Knopp et al. | AgentSimulator | **SimOC** |
|---------|-------------|----------------|-----------|
| Input format | OCEL 2.0 | XES | **OCEL 2.0** |
| Simulation paradigm | OCPN tokens | Multi-agent (resources) | **Multi-agent (object types)** |
| Agents represent | N/A | Resources/workers | **Objects (orders, items, packages)** |
| Multiple interacting types | Via OCPN arcs | No | **Yes (5 interaction patterns)** |
| Produces O2O relations | Yes | No | **Yes** |

## Installation

```bash
# Python 3.10+ required
git clone https://github.com/<your-username>/SimOC.git
cd SimOC
pip install -e ".[dev]"
```

## Quickstart

```bash
# Run on the OCEL 2.0 benchmark log
python -m simoc data/order-management.sqlite output/synthetic.jsonocel

# Multiple runs with different seeds
python -m simoc data/order-management.sqlite output/synthetic.jsonocel --runs 50

# Verbose output
python -m simoc data/order-management.sqlite output/synthetic.jsonocel -v
```

## Pipeline

SimOC runs a 7-stage pipeline: (1) OCEL 2.0 ingestion and validation, (2) object interaction graph construction and root/derived type classification, (3) per-type behavioral discovery (arrivals, durations, branching), (4) interaction pattern discovery (synchronization, binding, batching, release), (5) SimPy discrete-event simulation, (6) synthetic OCEL 2.0 export, (7) evaluation against baselines with statistical significance testing.

## Results

Comparison on Knopp's OCEL 2.0 Order Management benchmark (21K events, 10.8K objects, 10-run mean ± std, 30-day simulation). All SimOC vs baseline differences significant at p < 0.05 (Wilcoxon signed-rank, actual p ∈ [0.002, 0.004]).

| Metric | SimOC | Flat (single-type) | Independent |
|--------|-------|------------|-------------|
| Activity freq. EMD ↓ | **0.57 ± 0.01** | 1.57 ± 0.03 | 0.77 ± 0.03 |
| OC-DFG cosine sim. ↑ | 0.35 ± 0.01 | 0.15 ± 0.00 | **0.78 ± 0.03** |
| O2O fidelity ↑ | **0.15 ± 0.05** | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Cardinality KS ↑ | **0.13 ± 0.05** | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Conv./div. KS ↑ | **0.28 ± 0.01** | 0.00 ± 0.00 | 0.00 ± 0.00 |

**SimOC is the only method that produces O2O relational structure** (fidelity, cardinality, convergence/divergence > 0). Flat (single-type) and Independent baselines cannot produce O2O relations at all. Independent achieves higher OC-DFG similarity by replaying per-type DFGs without cross-type coordination — at the cost of zero structural fidelity.

## Tests

```bash
pytest tests/ -v    # 155 tests across all 7 stages
```

## Citation

```bibtex
@article{simoc2025,
  title     = {{SimOC}: Agent-Based Simulation of Object-Centric Processes
               Discovered from {OCEL} 2.0 Event Logs},
  author    = {<authors>},
  journal   = {Process Science},
  year      = {2025}
}
```

## License

MIT
