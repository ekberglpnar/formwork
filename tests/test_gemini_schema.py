"""The Gemini schema converter — offline, no SDK and no API key needed.

Worth its own file because every one of these assertions corresponds to a 400
the live run actually returned. Gemini's response_schema is an OpenAPI 3.0
subset, not JSON Schema, and Pydantic emits several things it rejects.
"""

from __future__ import annotations

import pytest

from formwork.providers.base import ProviderError
from formwork.providers.gemini import to_gemini_schema
from conftest import WorkoutPlan


@pytest.fixture
def schema():
    return to_gemini_schema(WorkoutPlan.model_facing_schema())


def test_additional_properties_is_stripped(schema):
    """The first live 400. Every formwork schema sets extra="forbid", so
    Pydantic emits additionalProperties, and Gemini rejects the whole request.
    """
    assert "additionalProperties" not in _keys(schema)


def test_refs_and_defs_are_inlined(schema):
    """Nested item models produce $ref/$defs and the API has no resolver."""
    assert "$defs" not in schema
    assert "$ref" not in _keys(schema)

    item = schema["properties"]["exercises"]["items"]
    assert item["type"] == "object"
    assert set(item["properties"]) == {"id", "sets"}


def test_pydantic_constraints_survive(schema):
    """These are the cheap guards worth keeping — they are enforced at decode."""
    sets = schema["properties"]["exercises"]["items"]["properties"]["sets"]
    assert sets == {"type": "integer", "minimum": 1, "maximum": 10}


def test_descriptions_survive(schema):
    assert schema["properties"]["exercises"]["description"] == "one entry per movement"


def test_required_and_type_survive(schema):
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"exercises", "rationale"}


def test_titles_are_dropped(schema):
    assert "title" not in _keys(schema)


def test_narrowed_schema_converts_too():
    """The repair path builds a fresh model per call; it must convert as well."""
    narrowed = to_gemini_schema(WorkoutPlan.model_facing_schema(only=("exercises",)))
    assert set(narrowed["properties"]) == {"exercises"}
    assert "additionalProperties" not in _keys(narrowed)


def test_unresolvable_reference_is_reported_clearly():
    from pydantic import BaseModel

    class Broken(BaseModel):
        x: int

    schema = {"$ref": "#/$defs/Missing"}
    Broken.model_json_schema = classmethod(lambda cls: dict(schema))  # type: ignore[assignment]

    with pytest.raises(ProviderError, match="unknown definition"):
        to_gemini_schema(Broken)


def _keys(node) -> set[str]:
    """Every key appearing anywhere in the tree."""
    found: set[str] = set()
    if isinstance(node, dict):
        found |= set(node)
        for value in node.values():
            found |= _keys(value)
    elif isinstance(node, list):
        for item in node:
            found |= _keys(item)
    return found
