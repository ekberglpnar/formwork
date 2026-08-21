"""Running the sweep.

Sequentially this takes hours, so runs go through a thread pool. The only
subtlety is temperature: at 0 every repetition of a cell returns the same
thing and N runs measure nothing. A success *rate* needs a distribution, so
the sweep runs at a realistic sampling temperature and repeats each cell.
"""

from __future__ import annotations

import itertools
import json
import random
import threading
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from formwork.providers.base import Model, ModelRequest, ProviderError
from formwork.report import Usage
from bench.arms import ARM_ORDER, RunOutcome, run_arm
from bench.errors import QuotaWall
from bench.tasks import DIFFICULTIES, TASKS

_TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "504")


@dataclass(frozen=True)
class Config:
    model: str
    temperature: float = 0.7
    runs: int = 12
    max_attempts: int = 3
    concurrency: int = 6
    tasks: Sequence[str] = tuple(TASKS)
    difficulties: Sequence[str] = DIFFICULTIES
    arms: Sequence[str] = tuple(ARM_ORDER)
    # 0 disables. On a free tier this is the difference between a result and
    # 300 rows of unusable data.
    requests_per_minute: float = 0.0


class RateLimiter:
    """Token bucket over requests per minute.

    Backoff alone is not enough. Once a per-minute quota is saturated, every
    worker discovers it independently, retries, and spends more quota finding
    out. Better to never exceed the rate in the first place.
    """

    def __init__(self, per_minute: float) -> None:
        self.interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if wait:
            time.sleep(wait)


class RetryingModel:
    """Rate-limits, backs off, and gives up loudly.

    Three attempts, not six: against a *daily* quota every extra retry is a
    request spent proving the wall is still there, and on a free tier that is
    the quota you wanted for the next arm.
    """

    def __init__(
        self,
        inner: Model,
        *,
        attempts: int = 3,
        base_delay: float = 2.0,
        per_minute: float = 0.0,
        wall_after: int = 25,
    ) -> None:
        self.inner = inner
        self.attempts = attempts
        self.base_delay = base_delay
        self.limiter = RateLimiter(per_minute)
        self.wall_after = wall_after
        self.throttled = 0
        self.consecutive_failures = 0
        self._lock = threading.Lock()

    def generate_structured(self, request: ModelRequest) -> tuple[dict[str, Any], Usage]:
        last: Exception | None = None
        for attempt in range(self.attempts):
            self._check_wall()
            self.limiter.acquire()
            try:
                result = self.inner.generate_structured(request)
            except ProviderError as error:
                if not any(token in str(error) for token in _TRANSIENT):
                    self._note_success()  # a real bug, not a quota problem
                    raise
                last = error
                with self._lock:
                    self.throttled += 1
                    self.consecutive_failures += 1
                delay = self.base_delay * (2**attempt) + random.uniform(0, 1.0)
                time.sleep(min(delay, 30.0))
            else:
                self._note_success()
                return result
        raise ProviderError(f"still throttled after {self.attempts} attempts: {last}")

    def _note_success(self) -> None:
        with self._lock:
            self.consecutive_failures = 0

    def _check_wall(self) -> None:
        with self._lock:
            hit = self.consecutive_failures >= self.wall_after
        if hit:
            raise QuotaWall(
                f"{self.consecutive_failures} consecutive throttled requests with no "
                "success in between — this is a hard quota, not a burst. Stopping so "
                "the partial results are not mistaken for a comparison."
            )


def jobs(config: Config) -> list[tuple[str, str, str, int]]:
    """Every cell of the sweep, shuffled.

    Shuffling matters: run arm by arm and any drift in the service over the
    hour lands entirely on whichever arms were unlucky.
    """
    combos = list(
        itertools.product(config.arms, config.tasks, config.difficulties, range(config.runs))
    )
    random.Random(0).shuffle(combos)
    return combos


def run_sweep(
    config: Config,
    model: Model,
    out_path: Path | str,
    *,
    on_progress: Any = None,
) -> list[RunOutcome]:
    retrying = RetryingModel(model, per_minute=config.requests_per_minute)
    todo = jobs(config)
    results: list[RunOutcome] = []
    aborted: QuotaWall | None = None

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    started = time.perf_counter()

    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"_config": {**asdict(config), "tasks": list(config.tasks),
                             "difficulties": list(config.difficulties),
                             "arms": list(config.arms)}},
                ensure_ascii=False,
            )
            + "\n"
        )

        with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
            futures = {
                pool.submit(
                    run_arm,
                    arm,
                    TASKS[task],
                    difficulty,
                    run_index,
                    retrying,
                    max_attempts=config.max_attempts,
                ): (arm, task, difficulty, run_index)
                for arm, task, difficulty, run_index in todo
            }

            for done, future in enumerate(as_completed(futures), start=1):
                try:
                    outcome = future.result()
                except QuotaWall as wall:
                    aborted = wall
                    for pending in futures:
                        pending.cancel()
                    break

                results.append(outcome)
                with write_lock:
                    handle.write(json.dumps(asdict(outcome), ensure_ascii=False) + "\n")
                    handle.flush()
                if on_progress:
                    on_progress(done, len(todo), outcome, time.perf_counter() - started)

    if aborted is not None:
        raise aborted

    return results


def load_results(path: Path | str) -> tuple[dict[str, Any], list[RunOutcome]]:
    config: dict[str, Any] = {}
    outcomes: list[RunOutcome] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "_config" in record:
            config = record["_config"]
            continue
        outcomes.append(RunOutcome(**record))
    return config, outcomes


def summarise_errors(outcomes: Iterable[RunOutcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        if outcome.error:
            key = outcome.error.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
    return counts
