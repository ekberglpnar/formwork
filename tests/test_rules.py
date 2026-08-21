"""Rule collection and the return shapes a predicate is allowed to use."""

from __future__ import annotations

from typing import Annotated

import pytest

from formwork import Spec, Violation, generated, rule, soft
from conftest import WorkoutPlan


class Shapes(Spec):
    value: Annotated[int, generated()]

    @rule("returned False")
    def by_bool(self, ctx):
        return self.value != 1

    @rule()
    def by_string(self, ctx):
        return "string failure" if self.value == 2 else None

    @rule()
    def by_iterable(self, ctx):
        if self.value == 3:
            return ["first", "second"]
        return None

    @rule()
    def by_violation(self, ctx):
        if self.value == 4:
            return Violation(rule="renamed", message="explicit", fields=("value",))
        return None

    @rule()
    def by_bad_type(self, ctx):
        return 3.14 if self.value == 5 else None


def test_false_uses_the_declared_message():
    report = Shapes(value=1).check(None)
    assert [v.message for v in report.violations] == ["returned False"]


def test_string_becomes_the_message():
    report = Shapes(value=2).check(None)
    assert [v.message for v in report.violations] == ["string failure"]


def test_iterable_yields_several_violations():
    report = Shapes(value=3).check(None)
    assert [v.message for v in report.violations] == ["first", "second"]
    assert all(v.rule == "by_iterable" for v in report.violations)


def test_explicit_violation_is_passed_through():
    report = Shapes(value=4).check(None)
    assert report.violations[0].rule == "renamed"
    assert report.violations[0].fields == ("value",)


def test_unusable_return_type_is_a_loud_error():
    with pytest.raises(TypeError, match="expected None, bool, str"):
        Shapes(value=5).check(None)


def test_passing_value_produces_nothing():
    assert Shapes(value=0).check(None).ok


def test_rules_are_collected_in_declaration_order():
    assert [r.name for r in WorkoutPlan.__formwork_rules__] == [
        "known_exercises",
        "not_too_many",
        "volume",
    ]


def test_objectives_are_collected_separately():
    assert [o.name for o in WorkoutPlan.__formwork_objectives__] == ["prefer_variety"]


def test_rules_are_inherited_and_extendable():
    class Extended(Shapes):
        @rule("extra")
        def another(self, ctx):
            return False

    names = [r.name for r in Extended.__formwork_rules__]
    assert "by_bool" in names and "another" in names


def test_implicated_fields_are_deduplicated():
    class Two(Spec):
        a: Annotated[int, generated()]

        @rule("x", fields=["a"])
        def one(self, ctx):
            return False

        @rule("y", fields=["a"])
        def two(self, ctx):
            return False

    assert Two(a=1).check(None).implicated_fields == ("a",)


def test_soft_score_is_weighted():
    class Scored(Spec):
        a: Annotated[int, generated()]

        @soft(weight=2.0)
        def cost(self, ctx):
            return 3.0

    total, breakdown = Scored(a=1).score(None)
    assert total == 6.0
    assert breakdown == {"cost": 6.0}
