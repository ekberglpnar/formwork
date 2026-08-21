"""Turning raw outcomes into numbers with error bars.

Rates get a Wilson interval rather than the textbook normal one, because at
these sample sizes the normal approximation misbehaves exactly where it
matters — near 0% and 100%, which is where several of these cells sit.

Token means are reported over *all* runs, failures included. Averaging only
the successes would flatter whichever arm fails most: it would get to drop its
expensive runs from the bill.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from bench.arms import ARM_ORDER
from bench.runner import RunOutcome

Z = 1.96  # 95%


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def pct(self) -> str:
        return f"{self.point * 100:.0f}% ({self.low * 100:.0f}-{self.high * 100:.0f})"

    def num(self, digits: int = 0) -> str:
        return f"{self.point:,.{digits}f} ±{(self.high - self.point):,.{digits}f}"


def wilson(successes: int, total: int) -> Interval:
    if total == 0:
        return Interval(0.0, 0.0, 0.0)
    p = successes / total
    denom = 1 + Z**2 / total
    centre = (p + Z**2 / (2 * total)) / denom
    margin = Z * math.sqrt(p * (1 - p) / total + Z**2 / (4 * total**2)) / denom
    return Interval(p, max(0.0, centre - margin), min(1.0, centre + margin))


def mean_ci(values: Sequence[float]) -> Interval:
    if not values:
        return Interval(0.0, 0.0, 0.0)
    mean = sum(values) / len(values)
    if len(values) == 1:
        return Interval(mean, mean, mean)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    se = math.sqrt(variance / len(values))
    return Interval(mean, mean - Z * se, mean + Z * se)


@dataclass
class ArmStats:
    arm: str
    n: int
    errors: int
    success: Interval
    valid_at_1: Interval
    tokens: Interval
    calls: Interval
    determ_repairs: float
    soft: float | None
    top_failures: list[tuple[str, int]]


def stats_for(arm: str, outcomes: Sequence[RunOutcome]) -> ArmStats:
    usable = [o for o in outcomes if o.error is None]
    errors = len(outcomes) - len(usable)

    successes = [o for o in usable if o.ok]
    failure_counts: dict[str, int] = {}
    for outcome in usable:
        if outcome.ok:
            continue
        for name in outcome.violated_rules or ["<none recorded>"]:
            failure_counts[name] = failure_counts.get(name, 0) + 1

    softs = [o.soft_score for o in successes if o.soft_score is not None]

    return ArmStats(
        arm=arm,
        n=len(usable),
        errors=errors,
        success=wilson(len(successes), len(usable)),
        valid_at_1=wilson(sum(1 for o in usable if o.valid_first_try), len(usable)),
        tokens=mean_ci([float(o.total_tokens) for o in usable]),
        calls=mean_ci([float(o.model_calls) for o in usable]),
        determ_repairs=(
            sum(o.deterministic_repairs for o in usable) / len(usable) if usable else 0.0
        ),
        soft=(sum(softs) / len(softs)) if softs else None,
        top_failures=sorted(failure_counts.items(), key=lambda kv: -kv[1])[:3],
    )


def by_arm(outcomes: Iterable[RunOutcome], arms: Sequence[str] | None = None) -> list[ArmStats]:
    grouped: dict[str, list[RunOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.arm, []).append(outcome)
    order = arms or [a for a in ARM_ORDER if a in grouped]
    return [stats_for(arm, grouped.get(arm, [])) for arm in order]


# ── comparisons that the arms were designed to support ───────────────────

# Each row differs from its predecessor in exactly one mechanism. Enforced by
# tests/test_bench_fairness.py, not by good intentions.
PAIRS = [
    ("Field ownership, one attempt", "single", "own-only"),
    ("Field ownership, with retry", "naive-retry", "own-retry"),
    ("Targeted vs full regeneration", "own-retry", "own-targeted"),
    ("Declared repairs", "own-targeted", "formwork"),
    ("Headline: hand-rolled vs formwork", "naive-retry", "formwork"),
]


#: A run is unusable if errors are this common, or this unevenly spread.
MAX_ERROR_RATE = 0.05
MAX_ERROR_SPREAD = 0.10


def balance_check(outcomes: Sequence[RunOutcome]) -> tuple[bool, str]:
    """Is the data fit to compare arms with?

    Dropped runs are the quiet way a benchmark lies. An arm that makes three
    calls has three chances to hit a rate limit where a one-shot arm has one,
    so throttling deletes the expensive arms' runs preferentially — and the
    runs it deletes are the *hard* ones, the ones that needed a repair. Both
    biases push in the same direction: they flatter whichever arm retries most,
    which here is the arm I wrote.

    So the report refuses rather than disclaims. A caveat under a table gets
    screenshotted away; a missing table does not.
    """
    per_arm: dict[str, list[RunOutcome]] = {}
    for outcome in outcomes:
        per_arm.setdefault(outcome.arm, []).append(outcome)
    if not per_arm:
        return False, "no runs"

    rates = {
        arm: sum(1 for o in runs if o.error) / len(runs) for arm, runs in per_arm.items()
    }
    worst = max(rates.values())
    spread = worst - min(rates.values())

    if worst <= MAX_ERROR_RATE and spread <= MAX_ERROR_SPREAD:
        return True, ""

    lines = [
        f"Errored runs reach {worst:.0%} and vary {spread:.0%} across arms "
        f"(thresholds: {MAX_ERROR_RATE:.0%} and {MAX_ERROR_SPREAD:.0%}).",
        "",
        "| Arm | runs | errored |",
        "|---|---|---|",
    ]
    for arm in sorted(rates, key=lambda a: -rates[a]):
        lines.append(f"| `{arm}` | {len(per_arm[arm])} | {rates[arm]:.0%} |")
    return False, "\n".join(lines)


def render(outcomes: Sequence[RunOutcome], config: dict[str, Any]) -> str:
    usable, detail = balance_check(outcomes)
    if not usable:
        return _refusal(outcomes, config, detail)
    return _render_full(outcomes, config)


def _refusal(outcomes: Sequence[RunOutcome], config: dict[str, Any], detail: str) -> str:
    return "\n".join(
        [
            "# formwork benchmark — NO RESULT",
            "",
            f"`{config.get('model', '?')}` · {len(outcomes)} runs recorded",
            "",
            "**This run is not reportable.** Too many runs failed for reasons that "
            "have nothing to do with the arms, and they did not fail evenly.",
            "",
            detail,
            "",
            "Runs lost to throttling are not lost at random. An arm that makes more "
            "calls has more chances to be cut off, and the runs cut off are "
            "disproportionately the ones that needed a repair — the hard ones. Both "
            "effects flatter the arms that retry most, which is the library's own arm.",
            "",
            "Fix the cause, then re-run:",
            "",
            "- `--rpm` to stay under the provider's per-minute quota",
            "- a paid tier, if the wall is a daily cap",
            "- fewer runs or fewer arms, if neither is available",
            "",
            "The raw rows are on disk and can be re-scored once the run is clean.",
        ]
    )


def _render_full(outcomes: Sequence[RunOutcome], config: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append

    total_tokens = sum(o.total_tokens for o in outcomes)
    add("# formwork benchmark")
    add("")
    add(
        f"`{config.get('model', '?')}` · temperature {config.get('temperature', '?')} · "
        f"{config.get('runs', '?')} runs per cell · max_attempts {config.get('max_attempts', '?')}"
    )
    add("")
    add(
        f"{len(outcomes)} runs across {len(config.get('tasks', []))} tasks × "
        f"{len(config.get('difficulties', []))} difficulties × "
        f"{len(config.get('arms', []))} arms · {total_tokens:,} tokens total"
    )
    add("")
    add("Rates carry 95% Wilson intervals. Token means include failed runs.")
    add("")

    add("## Overall")
    add("")
    add(_table(by_arm(outcomes)))
    add("")

    add("## What each mechanism is worth")
    add("")
    add("| Comparison | Success | valid@1 | Tokens |")
    add("|---|---|---|---|")
    lookup = {s.arm: s for s in by_arm(outcomes)}
    for label, before, after in PAIRS:
        if before not in lookup or after not in lookup:
            continue
        a, b = lookup[before], lookup[after]
        add(
            f"| {label} <br><sub>{before} → {after}</sub> "
            f"| {_delta_pct(a.success.point, b.success.point)} "
            f"| {_delta_pct(a.valid_at_1.point, b.valid_at_1.point)} "
            f"| {_delta_tokens(a.tokens.point, b.tokens.point)} |"
        )
    add("")

    tasks = sorted({o.task for o in outcomes})
    for task in tasks:
        subset = [o for o in outcomes if o.task == task]
        add(f"## Task: {task}")
        add("")
        add(_table(by_arm(subset)))
        add("")

        for difficulty in ("easy", "medium", "hard"):
            cell = [o for o in subset if o.difficulty == difficulty]
            if not cell:
                continue
            add(f"### {task} · {difficulty}")
            add("")
            add(_table(by_arm(cell), compact=True))
            add("")

    errors = [o for o in outcomes if o.error]
    if errors:
        add("## Runs that errored")
        add("")
        counts: dict[str, int] = {}
        for outcome in errors:
            counts[outcome.error or "?"] = counts.get(outcome.error or "?", 0) + 1
        for message, count in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
            add(f"- {count}× `{message[:160]}`")
        add("")

    return "\n".join(lines)


def _table(rows: Sequence[ArmStats], compact: bool = False) -> str:
    if compact:
        head = "| Arm | Success | valid@1 | Tokens | Calls |"
        sep = "|---|---|---|---|---|"
        body = [
            f"| `{r.arm}` | {r.success.pct()} | {r.valid_at_1.pct()} | "
            f"{r.tokens.point:,.0f} | {r.calls.point:.2f} |"
            for r in rows
        ]
        return "\n".join([head, sep, *body])

    head = "| Arm | n | Success | valid@1 | Tokens | Calls | Local fixes | Soft | Top failures |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    body = []
    for r in rows:
        failures = ", ".join(f"{name} ×{count}" for name, count in r.top_failures) or "—"
        soft = f"{r.soft:.2f}" if r.soft is not None else "—"
        body.append(
            f"| `{r.arm}` | {r.n} | {r.success.pct()} | {r.valid_at_1.pct()} | "
            f"{r.tokens.num()} | {r.calls.point:.2f} | {r.determ_repairs:.2f} "
            f"| {soft} | {failures} |"
        )
    return "\n".join([head, sep, *body])


def _delta_pct(before: float, after: float) -> str:
    delta = (after - before) * 100
    sign = "+" if delta >= 0 else ""
    return f"{before * 100:.0f}% → {after * 100:.0f}% ({sign}{delta:.0f} pts)"


def _delta_tokens(before: float, after: float) -> str:
    if before == 0:
        return f"{before:,.0f} → {after:,.0f}"
    change = (after - before) / before * 100
    sign = "+" if change >= 0 else ""
    return f"{before:,.0f} → {after:,.0f} ({sign}{change:.0f}%)"
