# Contributing to Formwork

Thanks for taking the time to look at Formwork. This document describes how to
set the project up, what the checks are, and which properties a change must not
break.

Formwork is early-stage, so the surface is still moving. If you are planning
something larger than a bug fix, open an issue or a
[discussion](https://github.com/ekberglpnar/formwork/discussions) first — it is
cheaper to agree on the shape of a change before it is written.

---

## Development setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

The only runtime dependency is Pydantic 2. Everything else — ruff, mypy, pytest
— comes from the `dev` extra. Formwork is developed and tested against Python
3.11, 3.12 and 3.13.

---

## Checks

Three commands must be clean before a pull request is ready. CI runs the same
three on every supported Python version, plus the end-to-end example.

```bash
.venv/bin/ruff check src tests examples bench scripts    # lint
.venv/bin/mypy                                           # strict; target is set in pyproject.toml
.venv/bin/python -m pytest -q                            # offline test suite
.venv/bin/python examples/workout.py                     # the example must still run
```

Useful subsets while iterating:

```bash
.venv/bin/python -m pytest tests/test_engine.py -q
.venv/bin/python -m pytest -q -k repair
```

### Live tests cost money

Tests marked `live` call a real provider API and are opt-in for that reason.
They are deselected by default and are not run in CI:

```bash
.venv/bin/python -m pytest -m live -q     # requires a provider key, spends credit
```

A change to a provider adapter should be verified against the live API at least
once before it is proposed, because adapters exist precisely to absorb the
differences between what a provider documents and what it accepts.

---

## Properties that must not break

These are the reasons the library exists. A change that weakens one of them
needs to argue for itself in the pull request description.

**`generate` never returns an object that violates a rule.** Either a valid
object or an exception. `tests/test_adversarial.py` asserts this against
deliberately hostile model behaviour across many seeds; treat that file as the
specification of the guarantee rather than as an ordinary test module.

**A deterministic repair counts as a failed attempt.** If a repair ran, the
model's own output was wrong, so `Attempt.ok` is `False`. `valid_first_try` is
the metric Formwork reports about itself — relaxing this would make the library
flatter its own benchmark.

**Loop logic lives only in `Session`.** `Session` is sans-IO: it emits requests
and consumes responses. `generate` and `agenerate` are thin drivers over it.
Logic added to a driver rather than to the state machine makes the synchronous
and asynchronous paths drift apart.

**The fake providers are public API.** `Chaos`, `Recording` and `Scripted` in
`formwork/providers/fake.py` are documented in the README so that users can
test their own specs. They are not internal test helpers, and their behaviour
should not change silently.

---

## Working on the pieces

**Field roles are metadata, not prompt text.** `computed()`, `chosen()` and
`generated()` return a `FieldSpec` that is read back out of the Pydantic field
metadata. To change what the model is allowed to invent, change the field's
role — not the wording of the prompt.

**New rules should declare `fields=`.** Targeted repair narrows the schema down
to the fields the violated rules point at and freezes the rest. A rule with no
declared fields is not an error; it silently falls back to regenerating
everything, which is the expensive path.

**Violation messages are read by the model.** They go straight into the repair
prompt, so they should name the concrete problem — `"ex_42 is not in the
library"` rather than `"invalid value"`.

---

## Style

- Code and documentation are in English.
- Comments explain *why* a decision was made, not *what* the line does.
- Match the surrounding code; ruff settings live in `pyproject.toml`.

---

## Pull requests

Keep the change focused, describe what it changes and why, and confirm that
lint, types, tests and the example are clean. If the change touches behaviour
covered by the properties above, say how you verified it.

By contributing, you agree that your contribution is licensed under the MIT
License, the same as the rest of the project.
