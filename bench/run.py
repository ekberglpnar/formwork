"""Benchmark entry point.

    .venv/bin/python -m bench.run --runs 12
    .venv/bin/python -m bench.run --report-only bench/results/run.jsonl

Costs money. Start with --runs 1 --tasks workout to see the bill.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from bench.arms import ARM_ORDER
from bench.errors import QuotaWall
from bench.report import render
from bench.runner import Config, load_results, run_sweep, summarise_errors
from bench.tasks import DIFFICULTIES, TASKS
from formwork.providers.base import ProviderError

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bench.run")
    parser.add_argument("--runs", type=int, default=12, help="repetitions per cell")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument(
        "--rpm",
        type=float,
        default=0.0,
        help="cap requests per minute (0 = no cap). Set this below the provider's "
        "quota; backoff alone will not save a free tier.",
    )
    parser.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    parser.add_argument(
        "--difficulties", nargs="*", default=list(DIFFICULTIES), choices=list(DIFFICULTIES)
    )
    parser.add_argument("--arms", nargs="*", default=list(ARM_ORDER), choices=list(ARM_ORDER))
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report-only", default=None, help="re-render an existing results file")
    return parser.parse_args(argv)


def progress(done: int, total: int, outcome, elapsed: float) -> None:
    rate = done / elapsed if elapsed else 0
    eta = (total - done) / rate if rate else 0
    flag = "!" if outcome.error else ("ok" if outcome.ok else "x ")
    sys.stderr.write(
        f"\r[{done:>5}/{total}] {flag} {outcome.arm:<12} {outcome.task:<9} "
        f"{outcome.difficulty:<6} eta {eta / 60:5.1f}m   "
    )
    sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.report_only:
        config, outcomes = load_results(args.report_only)
        print(render(outcomes, config))
        return 0

    load_dotenv(ROOT / ".env")
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is empty — fill in .env first.", file=sys.stderr)
        return 1

    from formwork.providers.gemini import DEFAULT_MODEL, Gemini

    model_name = args.model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    config = Config(
        model=model_name,
        temperature=args.temperature,
        runs=args.runs,
        max_attempts=args.max_attempts,
        concurrency=args.concurrency,
        tasks=tuple(args.tasks),
        difficulties=tuple(args.difficulties),
        arms=tuple(args.arms),
        requests_per_minute=args.rpm,
    )

    total = len(config.arms) * len(config.tasks) * len(config.difficulties) * config.runs
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out) if args.out else RESULTS / f"{stamp}.jsonl"

    print(f"model       : {model_name}")
    print(f"temperature : {config.temperature}")
    print(f"cells       : {len(config.arms)} arms × {len(config.tasks)} tasks × "
          f"{len(config.difficulties)} difficulties × {config.runs} runs = {total} runs")
    print(f"concurrency : {config.concurrency}")
    print(f"output      : {out_path}")
    print()

    model = Gemini(model=model_name, temperature=config.temperature)

    started = time.perf_counter()
    try:
        outcomes = run_sweep(config, model, out_path, on_progress=progress)
    except QuotaWall as wall:
        sys.stderr.write("\n")
        print(f"stopped: {wall}", file=sys.stderr)
        print(f"partial rows are in {out_path}; re-score with --report-only", file=sys.stderr)
        return 2
    except ProviderError as error:
        print(f"\nprovider error: {error}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    sys.stderr.write("\n")
    tokens = sum(o.total_tokens for o in outcomes)
    print(f"done in {elapsed / 60:.1f} minutes · {tokens:,} tokens")
    errors = summarise_errors(outcomes)
    if errors:
        print(f"errored runs: {errors}")
    print()

    report = render(outcomes, {**config.__dict__, "tasks": list(config.tasks),
                               "difficulties": list(config.difficulties),
                               "arms": list(config.arms)})
    report_path = out_path.with_suffix(".md")
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print()
    print(f"report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
