"""`biobrain` command line: download → preprocess → validate → analyze, plus lookups and benchmarks."""

from __future__ import annotations

import argparse
import sys

from . import paths
from .telemetry import DEFAULT_BUDGET, fmt_bytes, parse_bytes

DEFAULT_DATASET = "flywire_fafb_v783"


def cmd_datasets(args) -> int:
    from .connectome.catalog import list_catalogs, load_catalog

    for name in list_catalogs():
        cat = load_catalog(name)
        print(f"{cat.id}: {cat.title}\n  license: {cat.license}")
        for e in cat.files:
            print(f"  {e.id:<22}{fmt_bytes(e.size):>12}  {'required' if e.required else 'optional'}  {e.filename}")
    return 0


def cmd_download(args) -> int:
    from .connectome import download
    from .connectome.catalog import load_catalog

    return download.run(load_catalog(args.dataset), only=args.only, include_optional=args.include_optional,
                        allow_large=args.allow_large, max_file=parse_bytes(args.max_file_size), dry_run=args.dry_run,
                        retries=args.retries, parallel=args.parallel)


def cmd_preprocess(args) -> int:
    from .connectome import preprocess
    from .connectome.catalog import load_catalog
    from .telemetry import MemoryBudget

    budget = MemoryBudget.from_env(args.memory_budget)
    preprocess.run(load_catalog(args.dataset), budget=budget)
    print(f"peak RSS {fmt_bytes(budget.report()['peak_rss'])} (budget {fmt_bytes(budget.limit)})")
    return 0


def cmd_validate(args) -> int:
    from .connectome import validate
    from .connectome.catalog import load_catalog
    from .telemetry import MemoryBudget

    result = validate.run(load_catalog(args.dataset), budget=MemoryBudget.from_env(args.memory_budget))
    for c in result["checks"]:
        if c["status"] in ("FAIL", "WARN"):
            print(f"  {c['status']} {c['id']}: {c['description']} — observed {c['observed']}, expected {c['expected']}")
    return 0 if result["status"] == "PASS" else 1


def cmd_analyze(args) -> int:
    from .analysis import report
    from .analysis.analyzer import ConnectomeAnalyzer, Settings
    from .connectome.store import Connectome
    from .telemetry import MemoryBudget

    target = paths.results_dir() / "analysis" / args.dataset
    if args.render_only:
        print(f"report: {report.rerender(target)}")
        return 0
    budget = MemoryBudget.from_env(args.memory_budget)
    conn = Connectome.load(args.dataset)
    settings = Settings(seed=args.seed, path_sources=args.path_sources, betweenness_sources=args.betweenness_sources,
                        null_samples=args.null_samples, step_time_limit_s=args.step_time_limit)
    analyzer = ConnectomeAnalyzer(conn, settings, budget)
    out = analyzer.run()
    report.write(out, analyzer, conn, target)
    print(f"report: {target / 'REPORT.md'}")
    return 0


def cmd_neuron(args) -> int:
    import json

    from .analysis import lookup
    from .connectome.store import Connectome

    info = lookup.describe(Connectome.load(args.dataset), args.root_id, top=args.top)
    print(json.dumps(info, indent=1, default=str) if args.json else lookup.render(info))
    return 0


def cmd_bench_memory(args) -> int:
    from .benchmarks import memory

    cases = tuple(args.cases) if args.cases else memory.CASES
    out = memory.run(args.dataset, paths.results_dir() / "memory", cases=cases, seed=args.seed)
    failed = [r["case"] for r in out["results"] if "error" in r]
    print(f"results: {paths.results_dir() / 'memory' / 'memory_benchmark.json'}" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


def cmd_m2(args) -> int:
    from .snn import pipeline

    return pipeline.run_step(args.step)


def cmd_m25(args) -> int:
    from .snn import m25

    return m25.run_step(args.step)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biobrain", description=__doc__)
    parser.add_argument("--memory-budget", default=None,
                        help=f"RAM ceiling for heavy steps, binary units (default $BIOBRAIN_MEMORY_BUDGET or {DEFAULT_BUDGET})")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("datasets", help="list dataset catalogs and their files").set_defaults(func=cmd_datasets)

    p = sub.add_parser("download", help="download and verify the raw files listed in a catalog")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--only", nargs="+", metavar="FILE_ID", help="download just these catalog file ids")
    p.add_argument("--include-optional", action="store_true", help="also fetch files marked optional")
    p.add_argument("--allow-large", action="store_true", help="permit files flagged large or above --max-file-size")
    p.add_argument("--max-file-size", default="2GB")
    p.add_argument("--dry-run", action="store_true", help="show the plan (sizes, state, purpose) and stop")
    p.add_argument("--retries", type=int, default=8, help="retries per file on timeouts / HTTP 5xx (resumes)")
    p.add_argument("--parallel", type=int, default=3, help="files downloaded at the same time (one connection each)")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("preprocess", help="build the compact processed store (CSR/CSC + annotations + manifest)")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("validate", help="validate the store against invariants and published numbers")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("analyze", help="graph analysis report (results/analysis/<dataset>/REPORT.md)")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--path-sources", type=int, default=200)
    p.add_argument("--betweenness-sources", type=int, default=256)
    p.add_argument("--null-samples", type=int, default=5)
    p.add_argument("--step-time-limit", type=float, default=900, help="seconds; costlier steps are skipped or down-sampled")
    p.add_argument("--render-only", action="store_true", help="rebuild REPORT.md from an existing summary.json")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("bench-memory", help="benchmark graph representations, each in a fresh process")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--cases", nargs="+", help="subset of cases (default: all)")
    p.add_argument("--seed", type=int, default=20260916)
    p.set_defaults(func=cmd_bench_memory)

    p = sub.add_parser("m2", help="Milestone 2 spiking-engine experiments, one step at a time")
    p.add_argument("step", choices=["subgraphs", "calibrate", "calibrate-pb-glutamate", "long-equivalence", "baseline-regime", "equivalence", "baseline-rss", "bench", "bench-50k",
                                    "bench-neuropils", "patterns", "sensitivity", "nulls", "profile", "energy", "estimate", "figures", "summary"])
    p.set_defaults(func=cmd_m2)

    p = sub.add_parser("m25", help="Milestone 2.5: compiled backend and full-connectome validation, one step at a time")
    p.add_argument("step")
    p.set_defaults(func=cmd_m25)

    p = sub.add_parser("neuron", help="look up one neuron: annotations, degrees, strongest partners")
    p.add_argument("root_id", type=int)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_neuron)
    return parser


def main(argv: list[str] | None = None) -> int:
    paths.load_dotenv()
    sys.stdout.reconfigure(line_buffering=True)  # progress stays visible when redirected to a log
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
