"""Model adapters and the test doubles that stand in for them.

Real adapters live behind optional extras and are imported lazily, so that
``from formwork.providers import Chaos`` never drags in an SDK you have not
installed. Import ``Gemini`` from here and you get a clear error if the extra
is missing, rather than an ImportError from three frames down.
"""

from typing import TYPE_CHECKING, Any

from formwork.providers.base import AsyncModel, Model, ModelRequest, ProviderError
from formwork.providers.fake import (
    Always,
    AsyncAdapter,
    Chaos,
    Exhausted,
    Recording,
    Scripted,
)
from formwork.providers.record import JsonlRecorder, load_records, replay

if TYPE_CHECKING:
    from formwork.providers.gemini import AsyncGemini, Gemini

__all__ = [
    # protocol
    "Model",
    "AsyncModel",
    "ModelRequest",
    "ProviderError",
    # doubles
    "Scripted",
    "Always",
    "Chaos",
    "Recording",
    "AsyncAdapter",
    "Exhausted",
    # recording
    "JsonlRecorder",
    "replay",
    "load_records",
    # real adapters (lazy)
    "Gemini",
    "AsyncGemini",
]

_LAZY = {"Gemini": "formwork.providers.gemini", "AsyncGemini": "formwork.providers.gemini"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
