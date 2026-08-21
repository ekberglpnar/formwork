"""Recording live runs so they become offline fixtures.

The test doubles can invent wrong output, but they cannot invent the *shapes*
of wrongness a particular model actually produces. Every live run is therefore
worth keeping: wrap the real model, get a JSONL file, and replay it later for
free. Over time the replay corpus becomes the regression suite that ``Chaos``
cannot write for you.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from formwork.providers.base import Model, ModelRequest
from formwork.providers.fake import Scripted
from formwork.report import Usage

__all__ = ["JsonlRecorder", "replay", "load_records"]


@dataclass(slots=True)
class JsonlRecorder:
    """Wraps a model and appends every exchange to a JSONL file."""

    inner: Model
    path: Path | str
    include_prompt: bool = True
    _calls: int = field(default=0, init=False)

    def generate_structured(self, request: ModelRequest) -> tuple[dict[str, Any], Usage]:
        raw, usage = self.inner.generate_structured(request)
        self._append(request, raw, usage)
        self._calls += 1
        return raw, usage

    @property
    def call_count(self) -> int:
        return self._calls

    def _append(self, request: ModelRequest, raw: dict[str, Any], usage: Usage) -> None:
        record: dict[str, Any] = {
            "recorded_at": time.time(),
            "kind": request.kind,
            "fields": list(request.fields),
            "schema": request.schema.__name__,
            "response": raw,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "latency_ms": usage.latency_ms,
            },
        }
        if self.include_prompt:
            record["system"] = request.system
            record["prompt"] = request.prompt

        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(path: Path | str) -> list[dict[str, Any]]:
    """Read a recording back, in order."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def replay(path: Path | str) -> Scripted:
    """Turn a recording into a ``Scripted`` double.

    Token counts are carried over from the recording, so a replayed run reports
    the cost the live run actually incurred rather than the placeholder the
    fakes use.
    """
    records = load_records(path)
    if not records:
        raise ValueError(f"{path} has no records to replay")

    first = records[0]["usage"]
    return Scripted(
        responses=[record["response"] for record in records],
        usage=Usage(
            prompt_tokens=first["prompt_tokens"],
            completion_tokens=first["completion_tokens"],
            latency_ms=first["latency_ms"],
        ),
    )
