"""CLI entry point: python -m simoc <input> <output>"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="simoc",
        description="SimOC: Discover and simulate object-centric processes from OCEL 2.0 logs.",
    )
    parser.add_argument("input", help="Path to OCEL 2.0 event log (JSON-OCEL or SQLite)")
    parser.add_argument("output", help="Path for synthetic OCEL 2.0 output (.jsonocel or .sqlite)")
    parser.add_argument("--runs", type=int, default=1, help="Number of simulation runs (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--duration", type=float, default=None, help="Simulation duration in seconds (default: match real log)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from simoc.ingestion import load_ocel
    from simoc.discovery.interaction_graph import discover_interaction_graph
    from simoc.discovery.cardinality import compute_spawning_profiles
    from simoc.discovery.behavioral import discover_behavioral
    from simoc.discovery.patterns import discover_patterns
    from simoc.discovery.data_structures import DiscoveryResult
    from simoc.simulation.runner import SimulationRunner
    from simoc.output import export_ocel

    t0 = time.time()
    print(f"Loading {args.input}...")
    data = load_ocel(args.input)
    print(f"  {len(data.events)} events, {len(data.objects)} objects, {len(data.object_types)} types")

    print("Discovering process model...")
    bd, oig, tc = discover_interaction_graph(data)
    sp = compute_spawning_profiles(data, bd, tc)
    bp = discover_behavioral(data, bd, tc)
    ip = discover_patterns(data, bd, tc)
    dr = DiscoveryResult(bd, oig, tc, sp)

    # Determine simulation duration
    if args.duration:
        duration = args.duration
    else:
        ts = data.events["timestamp"]
        duration = (ts.max() - ts.min()).total_seconds()

    # Get start timestamp from real log
    start_ts = str(data.events["timestamp"].min())

    runner = SimulationRunner(dr, bp, ip)

    for i in range(args.runs):
        seed = args.seed + i
        print(f"Simulating run {i + 1}/{args.runs} (seed={seed})...")
        result = runner.run(duration=duration, seed=seed)
        print(f"  {len(result.events)} events, {len(result.objects)} objects")

        if args.runs == 1:
            out_path = args.output
        else:
            base = args.output.rsplit(".", 1)
            out_path = f"{base[0]}_run{i+1}.{base[1]}" if len(base) > 1 else f"{args.output}_run{i+1}"

        export_ocel(result, out_path, start_timestamp=start_ts)
        print(f"  Saved to {out_path}")

    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
