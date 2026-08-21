"""Google Gemini adapter.

Requires the optional extra::

    pip install "formwork[gemini]"

Two decisions in here are worth knowing about.

**The schema is passed to Gemini, but the response is still parsed raw.**
Gemini constrains the output to ``request.schema`` server-side, which is the
right thing — grammar-level enforcement is strictly better than checking after
the fact, and formwork has always claimed to compose with it rather than
replace it. But we hand the engine the undecoded dict anyway and let it
validate, so that a provider-side enforcement failure surfaces as an ordinary
structural error instead of an exception from inside the SDK.

**Thinking tokens are counted as completion tokens.** They are billed, so a
benchmark that omitted them would flatter every arm that thinks more.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from pydantic import BaseModel

from formwork.providers.base import ModelRequest, ProviderError
from formwork.report import Usage

__all__ = ["Gemini", "AsyncGemini", "DEFAULT_MODEL", "to_gemini_schema"]

# Model names age out — Google retires them for new projects and returns 404.
# Override with GEMINI_MODEL or the model= argument.
DEFAULT_MODEL = "gemini-3.5-flash"

# Gemini's response_schema is a subset of OpenAPI 3.0 Schema, not JSON Schema.
# Anything outside this set is rejected outright with a 400 — notably
# `additionalProperties`, which Pydantic emits for every model configured with
# `extra="forbid"`, i.e. every schema formwork builds.
_SUPPORTED_KEYS = frozenset(
    {
        "type",
        "format",
        "description",
        "nullable",
        "enum",
        "items",
        "properties",
        "required",
        "anyOf",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "pattern",
        "propertyOrdering",
    }
)


def to_gemini_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to something Gemini will actually accept.

    Two transformations. ``$ref``/``$defs`` are inlined, because the API has no
    reference resolver and formwork's nested item models always produce them.
    Then every key outside the supported subset is dropped.

    Dropping is safe in one direction only: the constraints Gemini keeps are
    enforced during decoding, and the ones it drops are still enforced by the
    engine when it validates the response. Losing `additionalProperties` here
    means an extra field comes back as a structural error instead of being
    prevented — a worse outcome, but a correct one.
    """
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})
    cleaned: dict[str, Any] = _clean(_inline(schema, definitions, depth=0))
    return cleaned


def _inline(node: Any, definitions: dict[str, Any], depth: int) -> Any:
    if depth > 32:
        raise ProviderError("schema nests deeper than 32 levels; refusing to inline")

    if isinstance(node, list):
        return [_inline(item, definitions, depth + 1) for item in node]
    if not isinstance(node, dict):
        return node

    if ref := node.get("$ref"):
        name = str(ref).rsplit("/", 1)[-1]
        if name not in definitions:
            raise ProviderError(f"schema references unknown definition {name!r}")
        merged = {**definitions[name], **{k: v for k, v in node.items() if k != "$ref"}}
        return _inline(merged, definitions, depth + 1)

    return {key: _inline(value, definitions, depth + 1) for key, value in node.items()}


def _clean(node: Any) -> Any:
    if isinstance(node, list):
        return [_clean(item) for item in node]
    if not isinstance(node, dict):
        return node

    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key not in _SUPPORTED_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {name: _clean(sub) for name, sub in value.items()}
        else:
            cleaned[key] = _clean(value)
    return cleaned


def _load_sdk() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ProviderError(
            "the Gemini adapter needs the google-genai SDK: "
            'pip install "formwork[gemini]"'
        ) from error
    return genai, types


class _Base:
    """Shared configuration and response handling."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
        client: Any = None,
    ) -> None:
        genai, types = _load_sdk()
        self._types = types

        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.temperature = temperature

        # Two knobs because the two model generations disagree. Gemini 2.5 took
        # thinking_budget, and 0 switched thinking off; the 3.x models reject a
        # budget of 0 outright with a bare "invalid argument" and want
        # thinking_level instead. Setting neither is fine and is the default.
        if thinking_budget is None:
            raw_budget = os.environ.get("GEMINI_THINKING_BUDGET")
            thinking_budget = int(raw_budget) if raw_budget not in (None, "") else None
        self.thinking_budget = thinking_budget

        self.thinking_level = thinking_level or os.environ.get("GEMINI_THINKING_LEVEL") or None

        if client is not None:
            self.client = client
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ProviderError(
                    "no API key: pass api_key=..., or set GEMINI_API_KEY "
                    "(copy .env.example to .env and fill it in)"
                )
            self.client = genai.Client(api_key=key)

    def _config(self, request: ModelRequest) -> Any:
        config = self._types.GenerateContentConfig(
            system_instruction=request.system,
            response_mime_type="application/json",
            response_schema=to_gemini_schema(request.schema),
            temperature=self.temperature,
        )
        if self.thinking_budget is not None or self.thinking_level is not None:
            kwargs: dict[str, Any] = {}
            if self.thinking_budget is not None:
                kwargs["thinking_budget"] = self.thinking_budget
            if self.thinking_level is not None:
                kwargs["thinking_level"] = self.thinking_level
            config.thinking_config = self._types.ThinkingConfig(**kwargs)
        return config

    def _decode(self, response: Any, elapsed_ms: int) -> tuple[dict[str, Any], Usage]:
        text = getattr(response, "text", None)
        if not text:
            # Usually a safety block or an empty candidate. Say which, because
            # "NoneType has no attribute" from three frames down is useless.
            reason = getattr(response, "prompt_feedback", None)
            raise ProviderError(f"Gemini returned no text (prompt_feedback={reason!r})")

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as error:
            raise ProviderError(
                f"Gemini returned non-JSON despite a schema: {text[:200]!r}"
            ) from error

        if not isinstance(raw, dict):
            raise ProviderError(f"expected a JSON object, got {type(raw).__name__}")

        return raw, self._usage(response, elapsed_ms)

    def _usage(self, response: Any, elapsed_ms: int) -> Usage:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return Usage(latency_ms=elapsed_ms)

        completion = _int(getattr(meta, "candidates_token_count", 0))
        completion += _int(getattr(meta, "thoughts_token_count", 0))
        return Usage(
            prompt_tokens=_int(getattr(meta, "prompt_token_count", 0)),
            completion_tokens=completion,
            latency_ms=elapsed_ms,
        )


class Gemini(_Base):
    """Synchronous Gemini model."""

    def generate_structured(self, request: ModelRequest) -> tuple[dict[str, Any], Usage]:
        started = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=request.prompt,
                config=self._config(request),
            )
        except Exception as error:  # noqa: BLE001 - re-raised below with context
            raise _wrap(error, self.model) from error
        return self._decode(response, _ms_since(started))


class AsyncGemini(_Base):
    """The same, awaited."""

    async def generate_structured(self, request: ModelRequest) -> tuple[dict[str, Any], Usage]:
        started = time.perf_counter()
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=request.prompt,
                config=self._config(request),
            )
        except Exception as error:  # noqa: BLE001 - re-raised below with context
            raise _wrap(error, self.model) from error
        return self._decode(response, _ms_since(started))


def _wrap(error: Exception, model: str) -> Exception:
    """Turn SDK errors into ProviderError, leaving anything else alone.

    A forty-frame tenacity traceback ending in a one-line 404 helps nobody, and
    the two failures worth naming are both routine: a retired model name and a
    schema the API would not take.
    """
    if isinstance(error, ProviderError):
        return error

    name = type(error).__name__
    if name not in {"ClientError", "ServerError", "APIError"}:
        return error

    detail = str(error)
    if "no longer available" in detail or "NOT_FOUND" in detail:
        return ProviderError(
            f"model {model!r} was rejected by the API. Model names get retired; "
            f"set GEMINI_MODEL to a current one. Original: {detail[:300]}"
        )
    if "response_schema" in detail:
        return ProviderError(
            "Gemini rejected the generated schema. This is an formwork bug — "
            f"to_gemini_schema() let through something unsupported. Original: {detail[:300]}"
        )
    return ProviderError(f"{name}: {detail[:400]}")


def _ms_since(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _int(value: Any) -> int:
    return int(value) if isinstance(value, int) else 0
