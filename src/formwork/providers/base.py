"""What the engine needs from a model, and nothing more.

Two methods' worth of surface, so that adapting a provider is a twenty-line
job and so that the test doubles in ``fake.py`` are honest stand-ins rather
than a parallel implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from formwork.report import Usage

__all__ = ["ModelRequest", "Model", "AsyncModel"]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One call the engine wants made."""

    prompt: str
    schema: type[BaseModel]
    system: str | None = None
    kind: str = "initial"  # "initial" | "repair"
    fields: tuple[str, ...] = ()

    @property
    def is_repair(self) -> bool:
        return self.kind == "repair"

    def json_schema(self) -> dict[str, Any]:
        """Convenience for providers that take a raw JSON Schema."""
        return self.schema.model_json_schema()


@runtime_checkable
class Model(Protocol):
    """A synchronous structured-output model."""

    def generate_structured(self, request: ModelRequest) -> tuple[dict[str, Any], Usage]:
        """Return the parsed JSON object and what it cost.

        Implementations should *not* validate against ``request.schema`` — the
        engine does that and needs the raw dict to write useful structural
        feedback when it fails.
        """
        ...


@runtime_checkable
class AsyncModel(Protocol):
    """The same, awaited."""

    async def generate_structured(self, request: ModelRequest) -> tuple[dict[str, Any], Usage]:
        ...
