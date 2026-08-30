# What this changes

<!-- What the change does, and why. Link the issue if there is one. -->

# Checks

- [ ] `ruff check src tests examples bench scripts`
- [ ] `mypy`
- [ ] `pytest -q`
- [ ] `python examples/workout.py`

# Behaviour

<!--
Delete this section if the change is documentation or tooling only.

If it touches the engine, the session, the rules or a provider adapter, say
which of the invariants in CONTRIBUTING.md it affects and how you verified it:

- generate never returns an object that violates a rule
- a deterministic repair counts as a failed attempt
- loop logic lives only in Session
- the fake providers are public API
-->
