# SimOC

Agent-based simulation discovery for object-centric processes from OCEL 2.0 event logs.

SimOC automatically discovers per-object-type agent behaviors and cross-type interaction patterns (spawning, synchronization, binding, batching, release) from real event data, then simulates forward to produce synthetic OCEL 2.0 traces.

| Feature | Knopp et al. (ICPM 2023) | AgentSimulator (ICPM 2024) | **SimOC** |
|---------|--------------------------|---------------------------|-----------|
| Discovers from event logs | Automated | Automated | Automated |
| Object-centric (OCEL) | OCEL 2.0 | XES only | **OCEL 2.0** |
| Agent-based architecture | OCPN tokens | Multi-agent | **Multi-agent** |
| Agents = object types | N/A | Agents = resources | **Yes** |
| Interaction patterns (spawn, sync, batch) | Partial (OCPN arcs) | No | **Yes** |
| Produces synthetic OCEL 2.0 | Yes | XES | **Yes** |

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

Comparison on Knopp's OCEL 2.0 Order Management benchmark (21K events, 10.8K objects, 5-run mean ± std, 30-day simulation):

| Metric | SimOC | Flat Simod | Independent |
|--------|-------|------------|-------------|
| Activity freq. EMD ↓ | **0.59 ± 0.05** | 1.53 ± 0.04 | 0.80 ± 0.04 |
| Arrival rate error ↓ | **0.25 ± 0.07** | 0.85 ± 0.00 | 0.60 ± 0.07 |
| OC-DFG cosine sim. ↑ | 0.43 ± 0.06 | 0.16 ± 0.00 | **0.78 ± 0.02** |
| O2O fidelity ↑ | **0.17 ± 0.00** | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Cardinality KS ↑ | **0.16 ± 0.00** | 0.00 ± 0.00 | 0.00 ± 0.00 |

SimOC is the only method that preserves object-to-object relational structure (O2O fidelity, cardinality) while maintaining competitive activity-level fidelity. Flat Simod and Independent baselines cannot produce O2O relations at all.

## Tests

```bash
pytest tests/ -v    # 155 tests across all 7 stages
```

## Citation

```bibtex
@inproceedings{simoc2026,
  title     = {{SimOC}: Agent-Based Simulation of Object-Centric Processes Discovered from {OCEL} 2.0 Event Logs},
  author    = {<authors>},
  booktitle = {International Conference on Process Mining (ICPM)},
  year      = {2026}
}
```

## License

MIT
